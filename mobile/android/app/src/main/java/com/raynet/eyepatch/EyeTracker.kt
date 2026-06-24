package com.raynet.eyepatch

import android.graphics.Bitmap

/**
 * Per-eye gaze orchestration (tier 2 + tier 3), the on-device analog of the
 * desktop `realtime_gaze.VOGEngine` loop for a single eye:
 *
 *   patch -> [EyeSegmenter] -> [SegPostProcess] -> [EllipseFitter] (pupil, iris)
 *         -> [EyeModel3D] (CALIBRATE: accumulate / TRACK: predict)
 *
 * State machine:
 *   IDLE        - segment + fit + draw, but no 3D model yet.
 *   CALIBRATING - accumulate confident pupil unprojections ("look around").
 *   TRACKING    - 3D eyeball fitted; emit gaze + pupil radius per frame.
 *
 * Each eye owns its own [EyeModel3D] (separate eyeball center/radius). The
 * [EyeSegmenter] is shared (one .ort, called sequentially on the pipeline thread).
 */
class EyeTracker(
    private val segmenter: EyeSegmenter,
    val side: String,                       // "left" / "right"
    focalPx: Double = 1200.0,
    pxPerMm: Double = 0.0,
    private val minConfidence: Double = 0.55,
) {
    enum class Mode { IDLE, CALIBRATING, TRACKING }

    private val model = EyeModel3D(focalPx = focalPx, pxPerMm = pxPerMm)
    var mode = Mode.IDLE; private set

    // Last chosen (smoothed) gaze normal, in the COMMON CAMERA FRAME. Drives
    // temporal continuity of the two-fold disambiguation and is published so the
    // OTHER eye can stay binocularly consistent (both eyes share this frame).
    @Volatile var currentGazeCam: EyeGeometry.V3? = null; private set
    // Last smoothed gaze in this eye's PATCH frame (for the on-patch overlay arrow).
    private var lastPatchGaze: EyeGeometry.V3? = null

    /** Everything the overlay needs for one frame. */
    class Result(
        val side: String,
        val mode: Mode,
        val clean: SegPostProcess.Clean?,
        val pupil: EllipseFitter.Fit?,
        val iris: EllipseFitter.Fit?,
        val gaze: EyeModel3D.Gaze?,
        val blink: Boolean,
        val blinkScore: Double,
        val calibCount: Int,
        val accepted: Boolean,              // pupil confident enough to use this frame
    )

    fun startCalibration() {
        model.reset(); currentGazeCam = null; lastPatchGaze = null; mode = Mode.CALIBRATING
    }

    /** Try to solve the eyeball model from accumulated frames. */
    fun finishCalibration(): Boolean {
        val ok = model.finishFit()
        currentGazeCam = null; lastPatchGaze = null
        mode = if (ok) Mode.TRACKING else Mode.IDLE
        return ok
    }

    fun reset() {
        model.reset(); currentGazeCam = null; lastPatchGaze = null; mode = Mode.IDLE
    }

    val calibCount get() = model.fitCount
    val isFitted get() = model.isFitted

    /**
     * @param rNorm    the patch's normalization rotation (camera->normalized), to
     *                 map this eye's gaze into the common camera frame.
     * @param flippedX true if the patch was horizontally mirrored (right + flip).
     * @param otherCam the OTHER eye's current camera-frame gaze (binocular hint).
     * @param rHead    head->camera rotation, for the head-pose prior.
     */
    fun process(
        patch: Bitmap,
        rNorm: Mat3,
        flippedX: Boolean,
        otherCam: EyeGeometry.V3? = null,
        rHead: Mat3? = null,
    ): Result {
        val seg = segmenter.segment(patch)
        val clean = SegPostProcess.process(seg)
        val pupil = EllipseFitter.fit(clean.pupilMask, clean.pupilProb, clean.w, clean.h)
        val iris = EllipseFitter.fit(clean.irisMask, clean.irisProb, clean.w, clean.h)
        val blinkScore = EllipseFitter.blinkScore(clean.pupilMask, clean.scleraMask)
        val blink = blinkScore.isNaN() || blinkScore < 0.735

        val accepted = pupil.valid && pupil.confidence >= minConfidence
        var gaze: EyeModel3D.Gaze? = null
        if (accepted) {
            val el = EyeModel3D.Ellipse(pupil.cx, pupil.cy, pupil.w, pupil.h, pupil.radian, pupil.confidence)
            when (mode) {
                Mode.CALIBRATING -> model.accumulate(el)
                Mode.TRACKING -> gaze = chooseGaze(el, rNorm, flippedX, otherCam, rHead)
                Mode.IDLE -> {}
            }
        }
        return Result(side, mode, clean, pupil.orNull(), iris.orNull(), gaze, blink, blinkScore,
            model.fitCount, accepted)
    }

    // Resolve the two-fold ambiguity in the COMMON CAMERA FRAME, scoring each
    // candidate by continuity + binocular agreement + the head-pose prior (see
    // [GazeConsistency]). currentGazeCam is published in that shared frame so the
    // other eye and the screen-gaze stage can use it directly; a light EMA on the
    // patch-frame gaze keeps the on-patch arrow steady.
    private fun chooseGaze(
        el: EyeModel3D.Ellipse, rNorm: Mat3, flippedX: Boolean,
        otherCam: EyeGeometry.V3?, rHead: Mat3?,
    ): EyeModel3D.Gaze? {
        val (a, b) = model.predictBoth(el) ?: return null
        val headFwd = if (rHead != null) GazeConsistency.headForward(rHead)
                      else EyeGeometry.V3(0.0, 0.0, -1.0)
        val aCam = patchToCam(a.gaze, rNorm, flippedX)
        val bCam = patchToCam(b.gaze, rNorm, flippedX)
        val sa = GazeConsistency.score(aCam, currentGazeCam, otherCam, headFwd)
        val sb = GazeConsistency.score(bCam, currentGazeCam, otherCam, headFwd)
        val chosen = if (sa >= sb) a else b

        // Light EMA: favour the new sample (0.65) to cut perceived lag at the low
        // ~4 fps gaze rate; the screen cursor gets extra IMU re-anchoring on top.
        val last = lastPatchGaze
        val smoothed = if (last == null) chosen.gaze
                       else (last * 0.35 + chosen.gaze * 0.65).normalized()
        lastPatchGaze = smoothed
        currentGazeCam = patchToCam(smoothed, rNorm, flippedX)
        return chosen.copy(gaze = smoothed)
    }

    /** Map a patch-frame direction to the camera frame: v_cam = rNorm^T * v. */
    private fun patchToCam(v: EyeGeometry.V3, rNorm: Mat3, flippedX: Boolean): EyeGeometry.V3 {
        val x = if (flippedX) -v.x else v.x
        val m = rNorm.m
        return EyeGeometry.V3(
            m[0] * x + m[3] * v.y + m[6] * v.z,
            m[1] * x + m[4] * v.y + m[7] * v.z,
            m[2] * x + m[5] * v.y + m[8] * v.z,
        ).normalized()
    }

    private fun EllipseFitter.Fit.orNull() = if (valid) this else null

    fun close() { /* segmenter is shared and closed by the pipeline */ }
}
