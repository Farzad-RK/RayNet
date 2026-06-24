package com.raynet.eyepatch

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer

/**
 * Head-pose inference with ONNX Runtime Mobile.
 *
 * Loads the `.ort` flatbuffer exported by `export_onnx.py --mobile` (static
 * batch=1, rotation-matrix-only output) and reproduces the EXACT preprocessing
 * from demo.py:  Resize(224) -> CenterCrop(224) -> ToTensor -> ImageNet-normalize.
 * Because the iris-anchored face crop is already square, Resize(224) maps it to
 * 224x224 and CenterCrop is a no-op — so we just scale the crop to 224x224.
 *
 * Output is the 3x3 head->camera rotation matrix (Mat3).
 */
class HeadPoseEstimator(context: Context, modelAsset: String = "head_pose_repnext_m4_mobile.ort") {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    private val side = 224
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std = floatArrayOf(0.229f, 0.224f, 0.225f)
    // Reused scratch buffer (NCHW: 1x3x224x224).
    private val chw = FloatArray(3 * side * side)
    private val pixels = IntArray(side * side)

    init {
        val modelBytes = context.assets.open(modelAsset).use { it.readBytes() }
        session = OrtAccel.createSession(env, modelBytes)
        inputName = session.inputNames.iterator().next()
    }

    /** @param faceCrop square iris-anchored crop (ARGB_8888). @return 3x3 rotation. */
    fun estimate(faceCrop: Bitmap): Mat3 {
        val scaled = Bitmap.createScaledBitmap(faceCrop, side, side, true)
        scaled.getPixels(pixels, 0, side, 0, 0, side, side)

        val plane = side * side
        for (i in 0 until plane) {
            val p = pixels[i]
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            chw[i] = (r - mean[0]) / std[0]               // R plane
            chw[plane + i] = (g - mean[1]) / std[1]        // G plane
            chw[2 * plane + i] = (b - mean[2]) / std[2]    // B plane
        }

        val shape = longArrayOf(1, 3, side.toLong(), side.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), shape).use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                val t = result[0] as OnnxTensor
                // Shape [1,3,3] -> flatten to 9 doubles, row-major.
                val flat = t.floatBuffer
                val m = DoubleArray(9) { flat.get(it).toDouble() }
                return Mat3(m)
            }
        }
    }

    fun close() {
        session.close()
        // env is process-global; do not close it here.
    }
}
