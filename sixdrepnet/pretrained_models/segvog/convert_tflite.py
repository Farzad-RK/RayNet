"""
Convert the SegResNet segmentation ONNX -> TFLite (NHWC, for the Android GPU
delegate) and validate output parity against onnxruntime.

Run with the isolated converter venv:
    sixdrepnet/.tflite_venv/bin/python \
        sixdrepnet/pretrained_models/segvog/convert_tflite.py
"""
import os, sys, glob
import numpy as np

# onnx2tf downloads a PICKLED calibration .npy; numpy 2 refuses it with
# allow_pickle=False. Patch np.load to allow pickle before importing onnx2tf.
_orig_load = np.load
np.load = lambda *a, **k: _orig_load(*a, **{**k, "allow_pickle": True})

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(HERE, "seg_segresnet_240x320_mobile.onnx")
OUTDIR = os.path.join(HERE, "tflite")

# 1. Convert with onnx2tf (NCHW onnx -> NHWC tflite, float32 + float16), in-process.
# onnx2tf wants to DOWNLOAD a sample image for its output-validation summary; the
# download is broken here, so stub it with random data of the expected shape.
print("== onnx2tf ==", flush=True)
import onnx2tf
import onnx2tf.onnx2tf as _o2t
_o2t.download_test_image_data = lambda: np.random.rand(20, 240, 320, 3).astype(np.float32)
onnx2tf.convert(input_onnx_file_path=ONNX, output_folder_path=OUTDIR, non_verbose=True)

tflites = sorted(glob.glob(os.path.join(OUTDIR, "*_float32.tflite")))
assert tflites, f"no *_float32.tflite produced in {OUTDIR}"
TFLITE = tflites[0]
print("tflite:", TFLITE, os.path.getsize(TFLITE), "bytes", flush=True)

# 2. Parity: same input through onnxruntime (NCHW) and tflite (NHWC).
import onnxruntime as ort
import tensorflow as tf

rng = np.random.default_rng(0)
# per-image min-max [0,1] grayscale repeated to 3 ch, like the device path
g = rng.random((240, 320), dtype=np.float32)
g = (g - g.min()) / (g.max() - g.min() + 1e-6)
nchw = np.stack([g, g, g], axis=0)[None].astype(np.float32)   # (1,3,240,320)
nhwc = np.transpose(nchw, (0, 2, 3, 1)).copy()                # (1,240,320,3)

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
onnx_out = sess.run(None, {sess.get_inputs()[0].name: nchw})[0]  # (1,3,240,320)

interp = tf.lite.Interpreter(model_path=TFLITE)
interp.allocate_tensors()
inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
print("tflite input", inp["shape"], "output", out["shape"], flush=True)
interp.set_tensor(inp["index"], nhwc)
interp.invoke()
tfl_out = interp.get_tensor(out["index"])                      # (1,240,320,3)

# align layouts -> (3,240,320)
onnx_p = onnx_out[0]
tfl_p = np.transpose(tfl_out[0], (2, 0, 1))
diff = np.abs(onnx_p - tfl_p)
print(f"parity max|onnx-tflite| = {diff.max():.3e}  mean = {diff.mean():.3e}", flush=True)
ok = diff.max() < 2e-3
print("PARITY", "OK" if ok else "FAIL (>2e-3)", flush=True)
sys.exit(0 if ok else 1)
