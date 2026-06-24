package com.raynet.eyepatch

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 3DeepVOG eye-feature SEGMENTATION via **TensorFlow Lite on the GPU delegate** —
 * the real-time replacement for the ONNX/CPU path (which was the frame-rate wall
 * at ~350 ms/inference; the Adreno GPU runs this conv-only net in ~tens of ms).
 *
 * The model is `seg_segresnet_240x320_mobile.tflite`, converted from the same
 * ONNX with `onnx2tf` (NCHW→**NHWC**, sigmoid baked in). Preprocessing mirrors
 * `deepvog3d_model.py::Model_3DeepVOG.predict`:
 *
 *   patch -> grayscale -> ScaleIntensity (per-image min-max to [0,1])
 *         -> repeat to 3 channels -> NHWC float32
 *
 * Output `seg` is NHWC (1,240,320,3) in [0,1] AFTER sigmoid; channel order
 * 0 = pupil, 1 = iris, 2 = sclera. Threshold at 0.5 downstream.
 *
 * Falls back to the multi-threaded CPU TFLite kernels if the GPU delegate is
 * unsupported on the device.
 */
class EyeSegmenter(
    context: Context,
    modelAsset: String = "seg_segresnet_240x320_mobile.tflite",
) {
    val w = 320
    val h = 240
    private val plane = w * h

    private val interpreter: Interpreter
    private var gpuDelegate: GpuDelegate? = null

    // Reused buffers.
    private var segNs = 0L; private var segCount = 0
    private val pixels = IntArray(plane)
    private val gray = FloatArray(plane)
    private val inBuf: ByteBuffer = ByteBuffer.allocateDirect(plane * 3 * 4).order(ByteOrder.nativeOrder())
    private val outBuf: ByteBuffer = ByteBuffer.allocateDirect(plane * 3 * 4).order(ByteOrder.nativeOrder())

    init {
        val bytes = context.assets.open(modelAsset).use { it.readBytes() }
        val model = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        model.put(bytes); model.rewind()

        val options = Interpreter.Options()
        val compat = CompatibilityList()
        if (compat.isDelegateSupportedOnThisDevice) {
            try {
                // fp16 compute roughly halves GPU conv time vs fp32 with negligible
                // mask accuracy loss (the seg is a soft mask thresholded at 0.5).
                val gpuOpts = compat.bestOptionsForThisDevice.apply { setPrecisionLossAllowed(true) }
                gpuDelegate = GpuDelegate(gpuOpts)
                options.addDelegate(gpuDelegate)
                Log.i("EyeSegmenter", "TFLite GPU delegate enabled (fp16)")
            } catch (e: Throwable) {
                Log.w("EyeSegmenter", "GPU delegate init failed: ${e.message}; CPU")
                options.setNumThreads(4)
            }
        } else {
            Log.i("EyeSegmenter", "GPU delegate unsupported; CPU (4 threads)")
            options.setNumThreads(4)
        }
        interpreter = Interpreter(model, options)
        val inShape = interpreter.getInputTensor(0).shape().joinToString("x")
        val outShape = interpreter.getOutputTensor(0).shape().joinToString("x")
        Log.i("EyeSegmenter", "tflite in=$inShape out=$outShape")
    }

    /** Segmentation probabilities for one 320x240 ARGB_8888 eye patch. */
    fun segment(patch: Bitmap): SegMaps {
        val src = if (patch.width == w && patch.height == h) patch
                  else Bitmap.createScaledBitmap(patch, w, h, true)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        // grayscale + per-image min/max
        var gMin = Float.MAX_VALUE; var gMax = -Float.MAX_VALUE
        for (i in 0 until plane) {
            val p = pixels[i]
            val lum = 0.299f * ((p shr 16) and 0xFF) + 0.587f * ((p shr 8) and 0xFF) + 0.114f * (p and 0xFF)
            gray[i] = lum
            if (lum < gMin) gMin = lum
            if (lum > gMax) gMax = lum
        }
        val inv = 1f / (gMax - gMin).let { if (it < 1e-6f) 1e-6f else it }

        // NHWC fill: [y][x][c], same value in all 3 channels.
        inBuf.rewind()
        for (i in 0 until plane) {
            val v = (gray[i] - gMin) * inv
            inBuf.putFloat(v); inBuf.putFloat(v); inBuf.putFloat(v)
        }
        inBuf.rewind(); outBuf.rewind()
        val t0 = android.os.SystemClock.elapsedRealtimeNanos()
        interpreter.run(inBuf, outBuf)
        segNs += android.os.SystemClock.elapsedRealtimeNanos() - t0
        if (++segCount >= 30) {
            Log.i("EyeSegmenter", "GPU seg avg = %.1f ms".format(segNs * 1e-6 / segCount))
            segNs = 0; segCount = 0
        }

        // NHWC read: deinterleave channels.
        outBuf.rewind()
        val pupil = FloatArray(plane); val iris = FloatArray(plane); val sclera = FloatArray(plane)
        for (i in 0 until plane) {
            pupil[i] = outBuf.float; iris[i] = outBuf.float; sclera[i] = outBuf.float
        }
        return SegMaps(pupil, iris, sclera, w, h)
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }
}
