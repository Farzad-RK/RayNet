package com.raynet.eyepatch

import android.graphics.Rect
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Iris-anchored square face crop — port of sixdrepnet/face_crop.py IrisFaceCropper.
 *
 * The crop is centered on the iris midpoint and sized by the 3D inter-ocular
 * distance (nearly constant under head rotation, so the crop does not zoom as the
 * head turns). This MUST match the training-time crop or the pose net drifts.
 */
class FaceCropper(
    private val sizeFactor: Double = 4.0,   // crop side = sizeFactor * IOD
    private val vertOffset: Double = 0.5,   // center this far below iris line (IOD units)
) {
    companion object {
        val LEFT_IRIS = intArrayOf(468, 469, 470, 471, 472)
        val RIGHT_IRIS = intArrayOf(473, 474, 475, 476, 477)
    }

    data class Result(
        val box: Rect,          // crop rectangle, clamped to frame
        val centerX: Int,       // pose-cube draw center
        val centerY: Int,
        val cube: Double,       // pose-cube size (px)
        val irisMidX: Double,   // iris-midpoint px (gaze-ray origin proxy)
        val irisMidY: Double,
        val iodPx: Double,      // rotation-invariant 3D inter-ocular distance (px)
    )

    /** landmarks: PIXEL units, each row [x, y, z] (z already scaled by width). */
    fun compute(landmarks: Array<DoubleArray>, w: Int, h: Int): Result? {
        if (landmarks.size < 478) return null  // need refined iris landmarks

        fun mean2(idx: IntArray): DoubleArray {
            var sx = 0.0; var sy = 0.0
            for (i in idx) { sx += landmarks[i][0]; sy += landmarks[i][1] }
            return doubleArrayOf(sx / idx.size, sy / idx.size)
        }
        fun mean3(idx: IntArray): DoubleArray {
            var sx = 0.0; var sy = 0.0; var sz = 0.0
            for (i in idx) { sx += landmarks[i][0]; sy += landmarks[i][1]; sz += landmarks[i][2] }
            return doubleArrayOf(sx / idx.size, sy / idx.size, sz / idx.size)
        }

        val l2 = mean2(LEFT_IRIS); val r2 = mean2(RIGHT_IRIS)
        val l3 = mean3(LEFT_IRIS); val r3 = mean3(RIGHT_IRIS)

        val dx = l3[0] - r3[0]; val dy = l3[1] - r3[1]; val dz = l3[2] - r3[2]
        val iod = maxOf(sqrt(dx * dx + dy * dy + dz * dz), 1e-3)

        val midX = 0.5 * (l2[0] + r2[0])
        val midY = 0.5 * (l2[1] + r2[1])
        val cx = midX
        val cy = midY + vertOffset * iod   // +y is down -> toward face center
        val half = 0.5 * sizeFactor * iod

        val box = squareBoxInFrame(cx, cy, half, w, h)
        return Result(box, cx.roundToInt(), cy.roundToInt(), 1.8 * iod, midX, midY, iod)
    }

    /** Largest square of given half-size centered near (cx,cy), kept in-frame. */
    private fun squareBoxInFrame(cxIn: Double, cyIn: Double, halfIn: Double, w: Int, h: Int): Rect {
        val half = minOf(halfIn, w / 2.0, h / 2.0)
        val cx = cxIn.coerceIn(half, w - half)
        val cy = cyIn.coerceIn(half, h - half)
        return Rect(
            (cx - half).roundToInt(), (cy - half).roundToInt(),
            (cx + half).roundToInt(), (cy + half).roundToInt(),
        )
    }
}
