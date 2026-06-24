package com.raynet.eyepatch

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker

/**
 * MediaPipe Face Landmarker wrapper — the on-device replacement for the desktop
 * MediaPipe FaceMesh. The bundled `face_landmarker.task` returns 478 landmarks
 * INCLUDING the iris ring (468-477), which our crop + eye-normalizer need.
 *
 * Runs in synchronous IMAGE mode, one face — simplest to drive from a CameraX
 * ImageAnalysis analyzer. (LIVE_STREAM async mode is a documented optimization.)
 */
class FaceLandmarkerHelper(context: Context, modelAsset: String = "face_landmarker.task") {

    private val landmarker: FaceLandmarker

    init {
        val base = BaseOptions.builder().setModelAssetPath(modelAsset).build()
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.IMAGE)
            .setNumFaces(1)
            .build()
        landmarker = FaceLandmarker.createFromOptions(context, options)
    }

    /**
     * Detect on [bitmap] but return coordinates in a TARGET pixel space. MediaPipe
     * landmarks are normalized [0,1], so we can detect on a small (fast) frame and
     * scale the coords to a high-res frame of the SAME aspect ratio — letting the
     * downstream crop/warp sample the high-res pixels (sharp eye patches) without
     * paying for landmarking at full resolution.
     *
     * @return landmarks in PIXEL units of (targetW, targetH), each row [x, y, z]
     *         with z scaled by targetW (matching face_crop.py), or null if no face.
     */
    fun detect(
        bitmap: Bitmap, targetW: Int = bitmap.width, targetH: Int = bitmap.height,
    ): Array<DoubleArray>? {
        val image = BitmapImageBuilder(bitmap).build()
        val result = landmarker.detect(image)
        val faces = result.faceLandmarks()
        if (faces.isEmpty()) return null
        val w = targetW.toDouble()
        val h = targetH.toDouble()
        val lms = faces[0]
        return Array(lms.size) { i ->
            val lm = lms[i]
            doubleArrayOf(lm.x() * w, lm.y() * h, lm.z() * w)
        }
    }

    fun close() = landmarker.close()
}
