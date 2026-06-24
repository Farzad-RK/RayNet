package com.raynet.eyepatch

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix as AndroidMatrix
import android.graphics.Paint

/**
 * Head-pose-normalized eye-patch extraction — a faithful port of
 * sixdrepnet/eye_norm.py (Zhang et al. 2018 data-normalization warp).
 *
 * The only platform difference: where the desktop uses cv2.warpPerspective, we
 * feed the same 3x3 homography W to android.graphics.Matrix + Canvas.drawBitmap.
 * Both apply W as a SOURCE->DEST forward map (OpenCV inverts internally for
 * sampling; Canvas maps source pixels onto the destination the same way), so W
 * is identical — no transpose/inverse juggling needed.
 *
 * Eyes are centered on the SOCKET (canthi midpoint), NOT the iris, so gaze stays
 * visible as pupil displacement inside the patch — what 3DeepVOG measures.
 */
class EyeNormalizer(
    private val outW: Int = 320,
    private val outH: Int = 240,
    private val fill: Double = 0.8,
    private val focal: Double? = null,
    private val flipRight: Boolean = false,
    private val smooth: Boolean = true,
    private val smoothMinCutoff: Double = 1.0,
    private val smoothBeta: Double = 0.3,
) {
    // MediaPipe FaceMesh canthi (eye corners). "right"/"left" are the subject's.
    companion object {
        const val R_OUT = 33; const val R_IN = 133      // right eye: outer, inner
        const val L_IN = 362; const val L_OUT = 263     // left eye:  inner, outer
        val ANCHORS = intArrayOf(R_OUT, R_IN, L_IN, L_OUT)
    }

    /** One Euro filter on a 2D pixel point (x and y independent). */
    private inner class PointSmoother {
        private val fx = OneEuroFilter(smoothMinCutoff, smoothBeta)
        private val fy = OneEuroFilter(smoothMinCutoff, smoothBeta)
        fun filter(px: Double, py: Double, t: Double) =
            doubleArrayOf(fx.filter(px, t), fy.filter(py, t))
    }

    private var anchorSm: MutableMap<Int, PointSmoother>? = null
    private var diagFrames = 0

    /**
     * A normalized eye patch plus the rotation used to produce it.
     *
     * `rNorm` (rows: camera->normalized) is the SAME matrix the warp applies, so a
     * gaze direction measured in this patch's virtual-camera frame maps back to the
     * real camera frame via `rNorm^T * v`. That common frame is what lets the two
     * eyes (each normalized in its OWN frame) be compared binocularly, and what the
     * head-pose prior is expressed in. `flippedX` is true when the patch was
     * horizontally mirrored (right eye + [flipRight]); the caller must negate the
     * gaze x-component before applying `rNorm^T`.
     */
    data class NormPatch(val bitmap: Bitmap, val rNorm: Mat3, val flippedX: Boolean)

    /** Drop anchor-smoothing state (call after a sustained face loss). */
    fun reset() { anchorSm = null }

    private fun intrinsics(w: Int, h: Int): Mat3 {
        val f = focal ?: w.toDouble()
        return Mat3(doubleArrayOf(
            f, 0.0, w / 2.0,
            0.0, f, h / 2.0,
            0.0, 0.0, 1.0,
        ))
    }

    private fun warpPt(H: Mat3, x: Double, y: Double): DoubleArray {
        val v = H.mul(Vec3(x, y, 1.0))
        return doubleArrayOf(v.x / v.z, v.y / v.z)
    }

    /** Returns the warped patch + its `Rnorm`, or null if the warp is degenerate. */
    private fun normOne(
        frame: Bitmap, K: Mat3, Kinv: Mat3, Rhead: Mat3,
        cu: Double, cv: Double,
        ax: Double, ay: Double, bx: Double, by: Double,
    ): Pair<Bitmap, Mat3>? {
        // 1. Viewing ray to the eye center (direction only).
        val ray = Kinv.mul(Vec3(cu, cv, 1.0))
        if (ray.norm() < 1e-9) return null
        val forward = ray.normalized()

        // 2. Normalized-camera rotation: look along `forward`, roll-leveled to the
        //    head x-axis (R_head column 0 = head's rightward axis in camera coords).
        val hRx = Rhead.col(0)
        var down = forward.cross(hRx)
        if (down.norm() < 1e-6) return null   // forward ∥ head-x (degenerate)
        down = down.normalized()
        val right = down.cross(forward).normalized()
        val Rnorm = Mat3.fromRows(right, down, forward)  // rows: camera->normalized

        // 3. Rotation homography into the virtual frontal/leveled view.
        val H = K * Rnorm * Kinv

        val c = warpPt(H, cu, cv)
        val a = warpPt(H, ax, ay)
        val b = warpPt(H, bx, by)
        val eyeW = kotlin.math.hypot(a[0] - b[0], a[1] - b[1])
        if (eyeW < 1e-3) return null

        // 4. Similarity: scale eye to `fill`*outW and center it in the patch.
        val scale = (fill * outW) / eyeW
        val S = Mat3(doubleArrayOf(
            scale, 0.0, outW / 2.0 - scale * c[0],
            0.0, scale, outH / 2.0 - scale * c[1],
            0.0, 0.0, 1.0,
        ))
        val W = S * H

        val out = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        val mtx = AndroidMatrix().apply { setValues(W.toFloat9()) }
        canvas.drawBitmap(frame, mtx, Paint(Paint.FILTER_BITMAP_FLAG))  // bilinear
        return Pair(out, Rnorm)
    }

    /**
     * @param frame      camera Bitmap (ARGB_8888), upright, full resolution.
     * @param landmarks  FaceMesh landmarks in PIXEL units; each row is [x, y, ...].
     * @param Rhead      head->camera rotation (use the SMOOTHED pose).
     * @param t          timestamp in seconds (drives anchor smoothing).
     * @return map possibly containing "left" / "right" -> patch Bitmap.
     */
    fun process(
        frame: Bitmap, landmarks: Array<DoubleArray>, Rhead: Mat3, t: Double,
    ): Map<String, NormPatch> {
        val w = frame.width; val h = frame.height
        val K = intrinsics(w, h)
        val Kinv = K.inverse()

        // Temporally smooth the canthi anchors (translation + scale jitter). The
        // gaze signal is the pupil INSIDE the socket, so smoothing the socket is
        // safe and does not blur gaze.
        val lm = Array(landmarks.size) { doubleArrayOf(landmarks[it][0], landmarks[it][1]) }
        if (smooth) {
            val sm = anchorSm ?: HashMap<Int, PointSmoother>().also { anchorSm = it }
            for (i in ANCHORS) {
                val p = sm.getOrPut(i) { PointSmoother() }.filter(lm[i][0], lm[i][1], t)
                lm[i][0] = p[0]; lm[i][1] = p[1]
            }
        }

        val out = HashMap<String, NormPatch>()

        // --- resolution diagnostic (throttled): how many SOURCE px land on the eye
        // before the warp scales it to fill*outW. upscale>1 => the patch is an
        // upsample of a smaller region (the cause of "smooth" patches).
        if (++diagFrames >= 30) {
            val canthi = kotlin.math.hypot(lm[R_OUT][0] - lm[R_IN][0], lm[R_OUT][1] - lm[R_IN][1])
            val upscale = if (canthi > 1e-3) (fill * outW) / canthi else 0.0
            android.util.Log.i("RayNetRes",
                "frame=%dx%d eyeSrcPx=%.0f -> patchEyePx=%.0f upscale=%.2fx (patch=%dx%d)"
                    .format(w, h, canthi, fill * outW, upscale, outW, outH))
            diagFrames = 0
        }

        // Right eye (subject's right; image-left).
        val crx = 0.5 * (lm[R_OUT][0] + lm[R_IN][0])
        val cry = 0.5 * (lm[R_OUT][1] + lm[R_IN][1])
        normOne(frame, K, Kinv, Rhead, crx, cry,
            lm[R_OUT][0], lm[R_OUT][1], lm[R_IN][0], lm[R_IN][1])?.let { (bmp, rNorm) ->
            out["right"] = NormPatch(if (flipRight) flipH(bmp) else bmp, rNorm, flipRight)
        }

        // Left eye (subject's left; image-right).
        val clx = 0.5 * (lm[L_OUT][0] + lm[L_IN][0])
        val cly = 0.5 * (lm[L_OUT][1] + lm[L_IN][1])
        normOne(frame, K, Kinv, Rhead, clx, cly,
            lm[L_OUT][0], lm[L_OUT][1], lm[L_IN][0], lm[L_IN][1])?.let { (bmp, rNorm) ->
            out["left"] = NormPatch(bmp, rNorm, false)
        }
        return out
    }

    private fun flipH(b: Bitmap): Bitmap {
        val m = AndroidMatrix().apply { preScale(-1f, 1f) }
        return Bitmap.createBitmap(b, 0, 0, b.width, b.height, m, true)
    }
}
