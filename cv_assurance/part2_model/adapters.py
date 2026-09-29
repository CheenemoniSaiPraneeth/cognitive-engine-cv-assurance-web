"""Module 1 - Model ingestion. Common adapter interface over NumPy-MLP (.npz), ONNX, TorchScript/PyTorch.

* Loading never executes arbitrary code: .npz uses allow_pickle=False, torch.load uses weights_only=True,
  ONNX with external data / custom operator domains is refused.
* Anything an adapter cannot do raises Unavailable(reason); the engine records UNAVAILABLE (never a silent PASS).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


class Unavailable(Exception):
    """A capability is not available for this artifact / environment."""


def preprocess(images, size, mean=0.5, std=0.25, layout="flat"):
    out = []
    for im in images:
        if im.shape[0] != size or im.shape[1] != size:
            im = np.asarray(Image.fromarray(im).resize((size, size), Image.BOX))
        out.append(im)
    x = (np.stack(out).astype(np.float32) / 255.0 - mean) / std
    if layout == "flat":
        return x.reshape(len(out), -1)
    if layout == "nchw":
        return x.transpose(0, 3, 1, 2).copy()
    return x  # nhwc


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class ModelAdapter:
    fmt = "abstract"
    white_box = False

    def __init__(self, path):
        self.path = Path(path)

    def load(self): raise NotImplementedError
    def metadata(self) -> dict: return {"format": self.fmt}
    def predict(self, images) -> np.ndarray: raise NotImplementedError
    def input_schema(self) -> dict: return {}
    def output_schema(self) -> dict: return {}
    def layers(self): raise Unavailable("layer inventory not available for this adapter/mode")
    def parameters(self): raise Unavailable("parameters not available for this adapter/mode")
    def activations(self, images): raise Unavailable("activations not available for this adapter/mode")
    def unit_influence(self): raise Unavailable("unit influence not available for this adapter/mode")
    def class_names(self): return None
    def close(self): pass


class NumpyMLPAdapter(ModelAdapter):
    """Reference white-box format used by the prototype: a .npz with W1,b1,W2,b2 and JSON meta."""
    fmt = "numpy-mlp"
    white_box = True

    def load(self):
        z = np.load(self.path, allow_pickle=False)
        self.p = {k: z[k].astype(np.float32) for k in ("W1", "b1", "W2", "b2")}
        self.meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        self.size = int(self.meta.get("input_size", 32))
        return self

    def metadata(self):
        return {"format": self.fmt, "architecture": f"MLP {self.p['W1'].shape[0]}-{self.p['W1'].shape[1]}-{self.p['W2'].shape[1]}",
                "parameter_count": int(sum(v.size for v in self.p.values())), "runtime": "numpy",
                "model_meta": self.meta}

    def input_schema(self): return {"dtype": "uint8", "layout": "HWC RGB", "resize_to": [self.size, self.size],
                                   "normalisation": [self.meta.get("mean", 0.5), self.meta.get("std", 0.25)]}
    def output_schema(self): return {"type": "classification", "num_classes": int(self.p["W2"].shape[1]), "activation": "softmax"}
    def class_names(self): return self.meta.get("class_names")

    def _x(self, images):
        return preprocess(images, self.size, self.meta.get("mean", 0.5), self.meta.get("std", 0.25))

    def predict(self, images):
        h = np.maximum(self._x(images) @ self.p["W1"] + self.p["b1"], 0)
        return softmax(h @ self.p["W2"] + self.p["b2"])

    def layers(self):
        return [{"name": "fc1", "type": "Linear", "shape": list(self.p["W1"].shape), "activation": "relu"},
                {"name": "fc2", "type": "Linear", "shape": list(self.p["W2"].shape), "activation": "softmax"}]

    def parameters(self):
        return {"fc1.weight": self.p["W1"], "fc1.bias": self.p["b1"], "fc2.weight": self.p["W2"], "fc2.bias": self.p["b2"]}

    def activations(self, images):
        h = np.maximum(self._x(images) @ self.p["W1"] + self.p["b1"], 0)
        return {"fc1_relu": h}

    def unit_influence(self):
        return {"fc1_relu": np.linalg.norm(self.p["W2"], axis=1)}


class OnnxAdapter(ModelAdapter):
    """ONNX via onnxruntime (predict) and, if installed, `onnx` (graph/initialisers = white-box).
    NOTE: written to the interface but NOT exercised in the prototype sandbox (no onnx/onnxruntime offline)."""
    fmt = "onnx"

    def load(self):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise Unavailable("onnxruntime not installed in this offline bundle") from e
        self._ort = ort
        self.sess = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        i = self.sess.get_inputs()[0]
        self.in_name, self.in_shape = i.name, i.shape
        self.layout = "nchw" if (len(i.shape) == 4 and i.shape[1] in (1, 3)) else "nhwc"
        dims = [d for d in i.shape[2:] if isinstance(d, int)] if self.layout == "nchw" else [d for d in i.shape[1:3] if isinstance(d, int)]
        self.size = dims[0] if dims else 224
        self._graph = None
        try:
            import onnx
            from onnx import numpy_helper
            m = onnx.load(str(self.path), load_external_data=False)
            if any(t.external_data for t in m.graph.initializer):
                raise Unavailable("ONNX uses external data files; refused for safe offline loading")
            std_domains = {"", "ai.onnx", "ai.onnx.ml"}
            bad = {n.domain for n in m.graph.node} - std_domains
            if bad:
                raise Unavailable(f"custom operator domains {sorted(bad)} not supported")
            self._graph = m
            self._params = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
            self.white_box = True
        except ImportError:
            pass
        return self

    def metadata(self):
        d = {"format": self.fmt, "runtime": f"onnxruntime {self._ort.__version__}", "input_shape": self.in_shape}
        if self._graph is not None:
            d["opset"] = [o.version for o in self._graph.opset_import]
            d["parameter_count"] = int(sum(v.size for v in self._params.values()))
            d["architecture"] = "ONNX graph: " + ",".join(sorted({n.op_type for n in self._graph.graph.node}))
        return d

    def input_schema(self): return {"name": self.in_name, "shape": self.in_shape, "layout": self.layout}
    def output_schema(self): return {"outputs": [o.name for o in self.sess.get_outputs()]}

    def predict(self, images):
        x = preprocess(images, self.size, 0.0, 1.0, self.layout)  # generic [0,1] scaling; override via wrapper if needed
        out = self.sess.run(None, {self.in_name: x})[0]
        out = np.asarray(out, dtype=np.float64).reshape(len(images), -1)
        if not np.allclose(out.sum(1), 1.0, atol=1e-3) or (out < 0).any():
            out = softmax(out)
        return out

    def layers(self):
        if self._graph is None: raise Unavailable("`onnx` package not installed: graph inspection unavailable")
        return [{"name": n.name or f"node{i}", "type": n.op_type,
                 "shape": [list(self._params[x].shape) for x in n.input if x in self._params]} for i, n in enumerate(self._graph.graph.node)]

    def parameters(self):
        if self._graph is None: raise Unavailable("`onnx` package not installed: initialisers unavailable")
        return self._params


class TorchAdapter(ModelAdapter):
    """TorchScript (predict + params) or plain state_dict (.pt/.pth loaded with weights_only=True: params only).
    NOTE: written to the interface but NOT exercised in the prototype sandbox (no torch offline)."""
    fmt = "torch"

    def __init__(self, path, size=224, class_count=None):
        super().__init__(path); self.size = size; self.mod = None; self.sd = None

    def load(self):
        try:
            import torch
        except ImportError as e:
            raise Unavailable("torch not installed in this offline bundle") from e
        self._t = torch
        try:
            self.mod = torch.jit.load(str(self.path), map_location="cpu").eval(); self.fmt = "torchscript"
            self.white_box = True
        except Exception:
            obj = torch.load(str(self.path), map_location="cpu", weights_only=True)
            if isinstance(obj, dict) and obj and all(hasattr(v, "shape") for v in obj.values()):
                self.sd = obj; self.fmt = "pytorch-state_dict"; self.white_box = True
            else:
                raise Unavailable("unsupported PyTorch artifact (needs TorchScript or a tensor state_dict)")
        return self

    def metadata(self):
        sd = self.parameters()
        return {"format": self.fmt, "runtime": f"torch {self._t.__version__}", "parameter_count": int(sum(v.size for v in sd.values())),
                "architecture": f"{len(sd)} tensors"}

    def predict(self, images):
        if self.mod is None: raise Unavailable("state_dict has no executable graph; behavioural tests unavailable")
        x = self._t.from_numpy(preprocess(images, self.size, 0.0, 1.0, "nchw"))
        with self._t.no_grad():
            out = self.mod(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
        out = out.numpy().astype(np.float64).reshape(len(images), -1)
        return softmax(out) if not np.allclose(out.sum(1), 1.0, atol=1e-3) else out

    def parameters(self):
        src = self.mod.state_dict() if self.mod is not None else self.sd
        return {k: v.detach().cpu().numpy() for k, v in src.items()}

    def layers(self):
        return [{"name": k, "type": "tensor", "shape": list(v.shape)} for k, v in self.parameters().items()]


class BlackBoxAdapter(ModelAdapter):
    """Wrap any adapter and expose ONLY predict() + file-level metadata -> forces graceful degradation."""
    white_box = False

    def __init__(self, inner):
        super().__init__(inner.path); self.inner = inner; self.fmt = inner.fmt + " (black-box view)"

    def load(self): return self
    def metadata(self): return {"format": self.fmt, "runtime": self.inner.metadata().get("runtime")}
    def predict(self, images): return self.inner.predict(images)
    def input_schema(self): return self.inner.input_schema()
    def output_schema(self): return self.inner.output_schema()
    def class_names(self): return self.inner.class_names()


def load_adapter(path, mode="auto", **kw):
    """mode: auto | blackbox. Returns a loaded adapter or raises Unavailable(reason)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".npz": a = NumpyMLPAdapter(p)
    elif ext == ".onnx": a = OnnxAdapter(p)
    elif ext in (".pt", ".pth", ".torchscript", ".ts"): a = TorchAdapter(p, **kw)
    else: raise Unavailable(f"unsupported model extension '{ext}'")
    a.load()
    return BlackBoxAdapter(a) if mode == "blackbox" else a
