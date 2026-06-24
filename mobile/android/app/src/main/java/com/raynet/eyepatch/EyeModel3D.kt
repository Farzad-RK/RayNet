package com.raynet.eyepatch

import com.raynet.eyepatch.EyeGeometry.V3
import kotlin.math.sqrt

/**
 * Per-eye single-sphere 3D eye model (DeepVOG / Świrski method), built on
 * [EyeGeometry]. Two phases, both fully on-device (Option A):
 *
 *  FIT (calibration "look-around"): accumulate per-frame unprojected pupils, then
 *  solve the least-squares intersection of their gaze lines for the eyeball
 *  CENTER and take the median pupil-to-center distance as the eyeball RADIUS.
 *
 *  PREDICT (live): unproject the pupil, disambiguate the two-fold normal against
 *  the fitted center, place the pupil on the eyeball, and emit gaze + pupil
 *  radius. Pupil radius comes from the unprojected-disk scale (the analog of
 *  pye3d's circle_3d.radius) — NOT the foreshortened 2D ellipse.
 *
 * ⚠️ The fit/predict ORCHESTRATION and the two-fold DISAMBIGUATION are
 * reconstructed from the DeepVOG method (the repo's active code uses pye3d, and
 * its own `simple`/`LeGrand` branches are stubs). Validate numerically against
 * the Python parity harness (sixdrepnet/3DeepVOG-main/tools/geom_parity.py)
 * before trusting absolute numbers. The primitives in [EyeGeometry] are direct
 * ports and are the trustworthy part.
 *
 * @param focalPx       effective focal length of the 320x240 patch, in px. This
 *                      must be DERIVED from the EyeNormalizer warp (see the
 *                      virtual-camera caveat in THREEDEEPVOG_DEPLOYMENT.md §7);
 *                      the placeholder default is for bring-up only.
 * @param assumedPupilRadiusPx  pupil-disk radius assumed during unprojection.
 * @param rEyeMm/rIrisMm        canonical eyeball/iris radii (DeepVOG defaults).
 * @param pxPerMm       patch px-per-mm, to report pupil radius in mm (optional).
 */
class EyeModel3D(
    private val focalPx: Double = 1200.0,
    private val assumedPupilRadiusPx: Double = 40.0,
    rEyeMm: Double = 12.0,
    rIrisMm: Double = 6.0,
    private val pxPerMm: Double = 0.0,
    private val cx: Double = 160.0,   // patch principal point (320/2)
    private val cy: Double = 120.0,   // (240/2)
) {
    // Pupil-to-eyeball-center distance, as a fraction of eyeball radius
    // (pupil sits on a smaller sphere): sqrt(r_eye^2 - r_iris^2) / r_eye.
    private val re2dp = sqrt(rEyeMm * rEyeMm - rIrisMm * rIrisMm) / rEyeMm

    // --- fit accumulators ---
    private val fitPos = ArrayList<V3>()
    private val fitNrm = ArrayList<V3>()

    // --- fitted model (null until finishFit succeeds) ---
    var eyeCenter: V3? = null; private set
    var eyeRadiusPx: Double = 0.0; private set
    val isFitted get() = eyeCenter != null

    /** One ellipse from [EllipseFit] (center-absolute patch px + radian). */
    data class Ellipse(
        val cx: Double, val cy: Double, val w: Double, val h: Double,
        val radian: Double, val confidence: Double,
    )

    /** Live per-frame output. */
    data class Gaze(
        val gaze: V3,                 // unit gaze direction (camera frame)
        val pupilCenter2D: DoubleArray,  // px in patch
        val eyeCenter2D: DoubleArray,    // px in patch
        val pupilRadiusPx: Double,
        val pupilRadiusMm: Double,    // 0 if pxPerMm not provided
        val confidence: Double,
    )

    fun reset() { fitPos.clear(); fitNrm.clear(); eyeCenter = null; eyeRadiusPx = 0.0 }

    // ---- FIT --------------------------------------------------------------
    /** Accumulate one calibration frame. Returns true if the frame was usable. */
    fun accumulate(el: Ellipse): Boolean {
        val un = unprojectChoose(el) ?: return false
        fitPos.add(un.center); fitNrm.add(un.normal)
        return true
    }

    /** Solve the eyeball model from accumulated frames. Returns true on success. */
    fun finishFit(): Boolean {
        if (fitPos.size < 8) return false
        val c = EyeGeometry.intersectLines(fitPos, fitNrm) ?: return false
        eyeCenter = c
        eyeRadiusPx = EyeGeometry.eyeRadius(fitPos, c)
        return eyeRadiusPx > 1e-6
    }

    val fitCount get() = fitPos.size

    // ---- PREDICT ----------------------------------------------------------
    fun predict(el: Ellipse): Gaze? {
        val c = eyeCenter ?: return null
        val un = unprojectChoose(el) ?: return null
        return buildGaze(c, un.normal, un.center, el)
    }

    /**
     * Both two-fold candidates (pos, neg) as full [Gaze]s, WITHOUT collapsing the
     * disambiguation. The caller resolves the ambiguity with cross-frame /
     * binocular continuity (the single-frame eyeball-outward test alone is too
     * weak — it made the two eyes disagree, one looking up and one down).
     */
    fun predictBoth(el: Ellipse): Pair<Gaze, Gaze>? {
        val c = eyeCenter ?: return null
        val ellCo = EyeGeometry.ellipseToGeneral(el.cx - cx, el.cy - cy, el.w, el.h, el.radian)
        val u = EyeGeometry.unproject(focalPx, ellCo, assumedPupilRadiusPx) ?: return null
        return Pair(buildGaze(c, u.normalPos, u.centerPos, el),
                    buildGaze(c, u.normalNeg, u.centerNeg, el))
    }

    private fun buildGaze(c: V3, normalRaw: V3, unprojCenter: V3, el: Ellipse): Gaze {
        val n = faceCamera(normalRaw.normalized())
        // Pupil center sits at re2dp * r_eye from the eyeball center, along gaze.
        val pupilDistPx = re2dp * eyeRadiusPx
        val pupilCenter3D = c + n * pupilDistPx
        // Pupil radius from the unprojected-disk scale: linear in depth.
        val unprojDepth = unprojCenter.z
        val radiusPx = if (unprojDepth > 1e-6)
            assumedPupilRadiusPx * (pupilCenter3D.z / unprojDepth) else 0.0
        return Gaze(
            gaze = n,
            pupilCenter2D = project(pupilCenter3D),
            eyeCenter2D = project(c),
            pupilRadiusPx = radiusPx,
            pupilRadiusMm = if (pxPerMm > 0) radiusPx / pxPerMm else 0.0,
            confidence = el.confidence,
        )
    }

    // ---- shared: unproject + two-fold disambiguation ----------------------
    private class Chosen(val normal: V3, val center: V3)

    /**
     * Orient a gaze normal to face the camera (−z). The Safaee-Rad unprojection
     * fixes the normal's LINE but not its SIGN; a visible pupil always faces the
     * camera, so the gaze points toward −z. Without this, ~half of gaze directions
     * came back inward-pointing (180° flipped), which also made the fit-time
     * z-comparison pick the wrong two-fold branch (validated by tools/tier3_parity).
     */
    private fun faceCamera(n: V3) = if (n.z > 0) n * -1.0 else n

    private fun unprojectChoose(el: Ellipse): Chosen? {
        // Center-relative conic (principal point at patch center).
        val ellCo = EyeGeometry.ellipseToGeneral(
            el.cx - cx, el.cy - cy, el.w, el.h, el.radian)
        val u = EyeGeometry.unproject(focalPx, ellCo, assumedPupilRadiusPx) ?: return null

        // SIGN first: both candidate normals must face the camera. Then resolve the
        // two-fold:
        //  - while fitting (no center yet): pick the more head-on branch (more
        //    negative z) — the mirror solution tilts further off-axis.
        //  - once fitted: pick the branch whose (pupil − eyeball centre) best aligns
        //    with its gaze normal (more physically consistent).
        val nPos = faceCamera(u.normalPos.normalized())
        val nNeg = faceCamera(u.normalNeg.normalized())
        val center = eyeCenter
        return if (center == null) {
            if (nPos.z <= nNeg.z) Chosen(nPos, u.centerPos)
            else Chosen(nNeg, u.centerNeg)
        } else {
            val dPos = (u.centerPos - center).normalized().dot(nPos)
            val dNeg = (u.centerNeg - center).normalized().dot(nNeg)
            if (dPos >= dNeg) Chosen(nPos, u.centerPos)
            else Chosen(nNeg, u.centerNeg)
        }
    }

    /** Reproject a camera-frame 3D point to patch px (pinhole, +z into scene). */
    private fun project(p: V3): DoubleArray {
        val z = if (p.z > 1e-6) p.z else 1e-6
        return doubleArrayOf(cx + focalPx * p.x / z, cy + focalPx * p.y / z)
    }
}
