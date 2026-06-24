package com.raynet.eyepatch

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Tier-2 ellipse fit — the analog of `threedeepvog/module/EllipseFitting.py`.
 *
 * DEVIATION (deliberate): the desktop fits `cv2.fitEllipse` to the blob
 * PERIMETER. We instead fit the **equivalent ellipse from image moments** of the
 * filled mask. For a filled, near-elliptical pupil/iris blob the two agree (both
 * recover the underlying disk), but moments need only a closed-form 2x2
 * eigendecomposition — no OpenCV AAR, no fragile Fitzgibbon eigen-solve, and it
 * is markedly more robust to ragged segmentation boundaries. The output
 * (cx, cy, w, h, radian) plugs straight into [EyeGeometry.ellipseToGeneral]
 * (== the desktop `convert_ell_to_general`), so the downstream geometry is
 * unchanged.
 *
 * Pure logic (no Android types) so it can be JVM-unit-tested.
 */
object EllipseFitter {

    /** Result in patch pixel coords: semi-axes (w along `radian`, h perpendicular). */
    data class Fit(
        val cx: Double, val cy: Double,     // center (px)
        val w: Double, val h: Double,        // semi-axes (px), w = major
        val radian: Double,                  // major-axis angle from +x (image frame)
        val confidence: Double,              // prob-mass fraction inside the ellipse
        val valid: Boolean,
    )

    private val INVALID = Fit(Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN, 0.0, false)

    /**
     * @param mask  cleaned boolean blob (row-major w*h) from [SegPostProcess].
     * @param prob  the source probability plane (for the confidence score).
     */
    fun fit(mask: BooleanArray, prob: FloatArray, w: Int, h: Int, minPixels: Int = 16): Fit {
        // --- raw + central second moments of the filled mask ---
        var m00 = 0.0; var m10 = 0.0; var m01 = 0.0
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) if (mask[row + x]) { m00 += 1.0; m10 += x; m01 += y }
        }
        if (m00 < minPixels) return INVALID
        val cx = m10 / m00; val cy = m01 / m00

        var mu20 = 0.0; var mu02 = 0.0; var mu11 = 0.0
        for (y in 0 until h) {
            val row = y * w; val dy = y - cy
            for (x in 0 until w) if (mask[row + x]) {
                val dx = x - cx
                mu20 += dx * dx; mu02 += dy * dy; mu11 += dx * dy
            }
        }
        mu20 /= m00; mu02 /= m00; mu11 /= m00

        // --- closed-form 2x2 symmetric eigendecomposition of [[mu20,mu11],[mu11,mu02]] ---
        val tr = mu20 + mu02
        val det = mu20 * mu02 - mu11 * mu11
        val disc = sqrt(maxOf(0.0, tr * tr / 4.0 - det))
        val l1 = tr / 2.0 + disc   // larger eigenvalue -> major axis
        val l2 = tr / 2.0 - disc
        if (l1 <= 1e-9) return INVALID

        // For a filled ellipse, central second moment along an axis = (semi-axis)^2 / 4.
        val semiMajor = 2.0 * sqrt(maxOf(l1, 0.0))
        val semiMinor = 2.0 * sqrt(maxOf(l2, 0.0))
        val radian = 0.5 * atan2(2.0 * mu11, mu20 - mu02)   // major-axis angle (image frame)
        if (semiMinor < 1e-3) return INVALID

        val conf = ellipseConfidence(prob, w, h, cx, cy, semiMajor, semiMinor, radian)
        return Fit(cx, cy, semiMajor, semiMinor, radian, conf, true)
    }

    /** Fraction of probability mass lying inside the fitted ellipse (port of
     *  EllipseConfidence_batch). */
    private fun ellipseConfidence(
        prob: FloatArray, w: Int, h: Int,
        cx: Double, cy: Double, a: Double, b: Double, theta: Double,
    ): Double {
        val ct = cos(theta); val st = sin(theta)
        // Only scan the axis-aligned bounding box of the ellipse.
        val rad = maxOf(a, b)
        val x0 = maxOf(0, (cx - rad).toInt()); val x1 = minOf(w - 1, (cx + rad).toInt() + 1)
        val y0 = maxOf(0, (cy - rad).toInt()); val y1 = minOf(h - 1, (cy + rad).toInt() + 1)
        var inside = 0.0; var mass = 0.0
        for (y in y0..y1) {
            val dy = y - cy
            for (x in x0..x1) {
                val dx = x - cx
                val xr = dx * ct + dy * st
                val yr = -dx * st + dy * ct
                if ((xr / a) * (xr / a) + (yr / b) * (yr / b) < 1.0) {
                    inside++; mass += prob[y * w + x]
                }
            }
        }
        return if (inside > 0) mass / inside else 0.0
    }

    /**
     * Blink score = |pupil ∩ sclera| / |pupil| (port of `blink_score` in
     * EllipseFitting.py). The desktop flags a blink when `score < blink_threshold`
     * (0.735). That inequality is counterintuitive and UNVALIDATED on our patches,
     * so [EyeTracker] treats blink as advisory (an annotation), not a hard gate —
     * gating is done on fit validity + confidence instead. Returns the raw score
     * (NaN if there is no pupil) so the caller can decide.
     */
    fun blinkScore(pupilMask: BooleanArray, scleraMask: BooleanArray): Double {
        var pupil = 0; var overlap = 0
        for (i in pupilMask.indices) if (pupilMask[i]) { pupil++; if (scleraMask[i]) overlap++ }
        if (pupil == 0) return Double.NaN
        return overlap.toDouble() / pupil
    }

    /** Faithful port: blink when score < threshold (advisory; see [blinkScore]). */
    fun isBlink(pupilMask: BooleanArray, scleraMask: BooleanArray, threshold: Double = 0.735): Boolean {
        val s = blinkScore(pupilMask, scleraMask)
        return s.isNaN() || s < threshold
    }
}
