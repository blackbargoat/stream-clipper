"""4x AI upscaling (Real-ESRGAN General x4v3, BSD-3, ONNX) for small facecams. Tiles of 128 px with overlap,
feather-blended so no seams show."""
import glob, os
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = glob.glob(os.path.join(HERE, "models", "resr", "**", "*.onnx"), recursive=True)[0]
_sess = None
T, OV = 128, 16


def _session():
    global _sess
    if _sess is None:
        opts = ort.SessionOptions(); opts.intra_op_num_threads = os.cpu_count() or 8
        _sess = ort.InferenceSession(_MODEL, opts, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    return _sess


def upscale4(rgb):
    """rgb uint8 HxWx3 -> uint8 (4H)x(4W)x3"""
    s = _session(); name = s.get_inputs()[0].name
    h, w = rgb.shape[:2]
    ys = list(range(0, max(1, h - T), T - OV)) + [max(0, h - T)]
    xs = list(range(0, max(1, w - T), T - OV)) + [max(0, w - T)]
    pad = np.pad(rgb, ((0, max(0, T - h)), (0, max(0, T - w)), (0, 0)), mode="edge")
    out = np.zeros((pad.shape[0] * 4, pad.shape[1] * 4, 3), np.float32); wt = np.zeros(out.shape[:2] + (1,), np.float32)
    ramp = np.minimum(1, (np.arange(T * 4) + 1) / (OV * 4)); ramp = np.minimum(ramp, ramp[::-1])
    feather = np.outer(ramp, ramp)[..., None].astype(np.float32)
    tiles, pos = [], []
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            tiles.append(pad[y:y + T, x:x + T].transpose(2, 0, 1).astype(np.float32) / 255.0); pos.append((y, x))
    for (y, x), t in zip(pos, tiles):
        r = s.run(None, {name: t[None]})[0][0].transpose(1, 2, 0)
        out[y * 4:(y + T) * 4, x * 4:(x + T) * 4] += r * feather; wt[y * 4:(y + T) * 4, x * 4:(x + T) * 4] += feather
    out = out / np.maximum(wt, 1e-6)
    return (np.clip(out[: h * 4, : w * 4], 0, 1) * 255).astype(np.uint8)
