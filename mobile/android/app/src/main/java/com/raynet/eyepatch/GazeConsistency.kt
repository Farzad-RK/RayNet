package com.raynet.eyepatch

import com.raynet.eyepatch.EyeGeometry.V3
import kotlin.math.cos

/**
 * Binocular + head-pose consistency for the two-fold gaze disambiguation.
 *
 * The unprojection of a pupil ellipse is two-fold ambiguous: two mirror gaze
 * normals reproject to the same ellipse. The single-frame "eyeball-outward" test
 * in [EyeModel3D] is too weak — it let the two eyes disagree (one looking up, the
 * other down). This resolver scores each camera-frame candidate against three
 * physical priors and the active eye picks the higher-scoring branch:
 *
 *   1. CONTINUITY  — gaze changes slowly, so prefer the branch near this eye's
 *      own previous (camera-frame) gaze.
 *   2. BINOCULAR   — the two eyes fixate a common point, so in the camera frame
 *      their gaze directions agree closely (parallel for far targets). Prefer the
 *      branch near the OTHER eye's current gaze.
 *   3. HEAD POSE   — the eye-in-head rotation is anatomically bounded (~±55°), so
 *      prefer the branch near head-forward and hard-reject any branch beyond the
 *      anatomical limit. Head-forward is the third column of the 6-DoF head
 *      rotation (its 6D representation is columns 0,1; column 2 = col0 × col1).
 *
 * All directions are unit vectors in the camera frame (origin at camera, +z INTO
 * the scene; a gaze looking back at the camera is ~(0,0,-1) — see [EyeGeometry]).
 */
object GazeConsistency {

    /** Anatomical eye-in-head rotation limit; branches beyond this are rejected. */
    const val MAX_EYE_ANGLE_DEG = 55.0
    private val maxCos = cos(Math.toRadians(MAX_EYE_ANGLE_DEG))

    private const val W_CONT = 1.0     // temporal continuity (this eye)
    private const val W_BINO = 0.8     // binocular agreement (other eye)
    private const val W_HEAD = 0.6     // head-pose forward prior
    private const val PENALTY = 10.0   // beyond the anatomical limit

    /**
     * Head-forward as an OUTGOING gaze direction in the camera frame (i.e. the
     * gaze of an eye looking straight ahead relative to the head).
     *
     * `rHead` is head->camera; column 2 is the head's forward axis in camera
     * coords. We orient it toward the camera (-z) to match the gaze convention.
     * If a future head-pose model uses a different axis convention, this is the
     * single line to flip.
     */
    fun headForward(rHead: Mat3): V3 {
        val f = rHead.col(2)
        var v = V3(f.x, f.y, f.z).normalized()
        if (v.z > 0) v = v * -1.0
        return v
    }

    /**
     * Consistency score for one camera-frame gaze candidate. Higher is better.
     * @param lastSelf this eye's previous camera-frame gaze (null on first frame).
     * @param other    the other eye's current camera-frame gaze (null if unknown).
     * @param headFwd  [headForward] of the current head pose.
     */
    fun score(cand: V3, lastSelf: V3?, other: V3?, headFwd: V3): Double {
        val c = cand.normalized()
        var s = W_HEAD * c.dot(headFwd)
        if (lastSelf != null) s += W_CONT * c.dot(lastSelf.normalized())
        if (other != null) s += W_BINO * c.dot(other.normalized())
        if (c.dot(headFwd) < maxCos) s -= PENALTY      // anatomically impossible
        return s
    }
}
