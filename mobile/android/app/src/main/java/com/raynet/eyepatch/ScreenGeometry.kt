package com.raynet.eyepatch

/**
 * Metric gaze->screen geometry (the geometric half of the hybrid mapping).
 *
 *  1. BACK-PROJECT the eye to a 3D point in the camera frame, with depth from the
 *     inter-ocular distance:  Z = f * IOD_mm / IOD_px. The lateral X,Y are actually
 *     focal-INDEPENDENT — (u-cx)*IOD_mm/IOD_px — so only depth rides on the assumed
 *     focal length, which keeps the estimate robust.
 *  2. INTERSECT the gaze ray with the device's screen plane (z = 0; the camera sits
 *     on it, looking out along +z toward the user).
 *  3. EMBED the camera-frame mm hit-point into the calibration canvas, using the
 *     physical screen size (DisplayMetrics) and the camera's top-center position.
 *
 * The absolute screen embedding (camera offset, axis sign/scale) is only
 * approximate — it's the per-user learned residual in [GazeScreenCalibrator] that
 * removes the linear part (and the kappa angle). Geometry's job is the parts the
 * residual can't learn from a fixed pose: perspective/depth scaling and (via the
 * IMU delta) device tilt. Everything in mm unless noted.
 *
 * @param viewWidthPx/viewHeightPx   the calibration canvas size (px).
 * @param viewLeftPx/viewTopPx       canvas top-left in the display's coordinates.
 * @param pxPerMmX/pxPerMmY          display density (DisplayMetrics xdpi/ydpi / 25.4).
 * @param camXpx/camYpx             front-camera position in display coords (top-center).
 * @param focalFactor               camera focal length as a multiple of the frame width.
 */
class ScreenGeometry(
    private val viewWidthPx: Int,
    private val viewHeightPx: Int,
    private val viewLeftPx: Int,
    private val viewTopPx: Int,
    private val pxPerMmX: Double,
    private val pxPerMmY: Double,
    private val camXpx: Double,
    private val camYpx: Double,
    private val focalFactor: Double = 1.0,
) {
    companion object { const val IOD_MM = 63.0 }   // adult mean inter-ocular distance

    /**
     * @param eyeU/eyeV  iris-midpoint pixel in the full-res frame.
     * @param iodPx      inter-ocular distance in px (rotation-invariant 3D estimate).
     * @param frameW/H   full-res frame dimensions.
     * @param gaze       combined gaze direction in the camera frame (z<0 = toward screen).
     * @param deltaR     device-frame rotation delta from the IMU (identity if none) —
     *                   re-anchors a world-fixed gaze into the CURRENT device frame.
     * @return normalized canvas point [nx, ny] (may fall outside [0,1]), or null if
     *   the ray does not cross the screen plane.
     */
    fun project(
        eyeU: Double, eyeV: Double, iodPx: Double,
        frameW: Int, frameH: Int,
        gaze: EyeGeometry.V3, deltaR: Mat3? = null,
    ): DoubleArray? {
        if (iodPx < 1e-3) return null
        val f = focalFactor * frameW
        val cx0 = frameW / 2.0; val cy0 = frameH / 2.0
        val z = f * IOD_MM / iodPx                       // depth (mm)
        var eye = Vec3((eyeU - cx0) * IOD_MM / iodPx, (eyeV - cy0) * IOD_MM / iodPx, z)
        var g = Vec3(gaze.x, gaze.y, gaze.z).normalized()

        if (deltaR != null) { eye = deltaR.mul(eye); g = deltaR.mul(g) }
        if (g.z > -1e-4) return null                     // not looking toward the screen
        val t = -eye.z / g.z
        val hitX = eye.x + g.x * t                        // mm, camera-frame
        val hitY = eye.y + g.y * t

        // Camera-frame mm -> display px -> canvas-normalized.
        val sx = camXpx + hitX * pxPerMmX
        val sy = camYpx + hitY * pxPerMmY
        return doubleArrayOf(
            (sx - viewLeftPx) / viewWidthPx,
            (sy - viewTopPx) / viewHeightPx,
        )
    }
}
