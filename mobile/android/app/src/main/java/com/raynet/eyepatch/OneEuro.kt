package com.raynet.eyepatch

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * One Euro filter + quaternion smoother — a faithful port of sixdrepnet/filters.py.
 *
 * The desktop pipeline smooths the head rotation in quaternion space (never in
 * Euler — quaternions avoid wraparound/gimbal lock) and, separately, smooths the
 * eye-patch anchor points. This file provides both the scalar filter and the
 * quaternion wrapper plus rotation<->quaternion conversion.
 *
 * Timestamps `t` are in SECONDS. On device, pass a monotonic clock
 * (e.g. SystemClock.elapsedRealtimeNanos() * 1e-9).
 *
 * NOTE: the desktop HeadPoseSmoother also runs a robust soft-median sliding
 * window IN FRONT of this One Euro stage (filters.py: WeightedWindowQuaternion).
 * For the "simplest working demo" we ship the One Euro stage only; it already
 * removes the bulk of the jitter. Porting the window is a documented follow-up.
 */

/** First-order exponential low-pass with externally supplied alpha. */
private class LowPass {
    private var y: Double? = null
    fun filter(x: Double, alpha: Double): Double {
        val prev = y
        val out = if (prev == null) x else alpha * x + (1.0 - alpha) * prev
        y = out
        return out
    }
}

/** Scalar One Euro filter (Casiez et al., CHI 2012). */
class OneEuroFilter(
    private val minCutoff: Double = 1.0,
    private val beta: Double = 0.3,
    private val dCutoff: Double = 1.0,
) {
    private val xLp = LowPass()
    private val dxLp = LowPass()
    private var xPrev: Double? = null
    private var tPrev: Double? = null

    private fun alpha(cutoff: Double, dt: Double): Double {
        val tau = 1.0 / (2.0 * PI * cutoff)
        return 1.0 / (1.0 + tau / dt)
    }

    fun filter(x: Double, t: Double): Double {
        val tp = tPrev
        if (tp == null) {
            tPrev = t; xPrev = x
            return x
        }
        var dt = t - tp
        if (dt <= 0) dt = 1e-3  // guard non-monotonic timestamps
        val dx = (x - (xPrev ?: x)) / dt
        val dxHat = dxLp.filter(dx, alpha(dCutoff, dt))
        val cutoff = minCutoff + beta * abs(dxHat)
        val xHat = xLp.filter(x, alpha(cutoff, dt))
        xPrev = x; tPrev = t
        return xHat
    }
}

/** Quaternion (w, x, y, z) as DoubleArray(4). */
typealias Quat = DoubleArray

/** 3x3 rotation matrix -> unit quaternion (w, x, y, z). Port of rotmat_to_quat. */
fun rotmatToQuat(R: Mat3): Quat {
    val tr = R[0, 0] + R[1, 1] + R[2, 2]
    val q: Quat
    if (tr > 0.0) {
        val s = 0.5 / sqrt(tr + 1.0)
        q = doubleArrayOf(
            0.25 / s,
            (R[2, 1] - R[1, 2]) * s,
            (R[0, 2] - R[2, 0]) * s,
            (R[1, 0] - R[0, 1]) * s,
        )
    } else if (R[0, 0] > R[1, 1] && R[0, 0] > R[2, 2]) {
        val s = 2.0 * sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = doubleArrayOf(
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        )
    } else if (R[1, 1] > R[2, 2]) {
        val s = 2.0 * sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = doubleArrayOf(
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        )
    } else {
        val s = 2.0 * sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = doubleArrayOf(
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        )
    }
    val n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return doubleArrayOf(q[0] / n, q[1] / n, q[2] / n, q[3] / n)
}

/** Unit quaternion (w, x, y, z) -> 3x3 rotation matrix. Port of quat_to_rotmat. */
fun quatToRotmat(q: Quat): Mat3 {
    val (w, x, y, z) = q
    return Mat3(doubleArrayOf(
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
    ))
}

/**
 * One Euro filter on a unit quaternion: each component filtered independently,
 * input hemisphere-aligned to the previous sample (double cover), output
 * renormalized. Port of OneEuroQuaternion.
 */
class OneEuroQuaternion(
    minCutoff: Double = 1.0,
    beta: Double = 0.3,
    dCutoff: Double = 1.0,
) {
    private var filters = Array(4) { OneEuroFilter(minCutoff, beta, dCutoff) }
    private var qPrev: Quat? = null
    private val mc = minCutoff
    private val b = beta
    private val dc = dCutoff

    fun reset() {
        filters = Array(4) { OneEuroFilter(mc, b, dc) }
        qPrev = null
    }

    fun filter(qIn: Quat, t: Double): Quat {
        var q = qIn
        val n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if (n < 1e-12) return q
        q = doubleArrayOf(q[0] / n, q[1] / n, q[2] / n, q[3] / n)

        val prev = qPrev
        if (prev != null) {
            val dot = q[0] * prev[0] + q[1] * prev[1] + q[2] * prev[2] + q[3] * prev[3]
            if (dot < 0.0) q = doubleArrayOf(-q[0], -q[1], -q[2], -q[3])
        }
        qPrev = q

        val qHat = DoubleArray(4) { filters[it].filter(q[it], t) }
        val nh = sqrt(qHat[0] * qHat[0] + qHat[1] * qHat[1] + qHat[2] * qHat[2] + qHat[3] * qHat[3])
        if (nh < 1e-12) return q
        return doubleArrayOf(qHat[0] / nh, qHat[1] / nh, qHat[2] / nh, qHat[3] / nh)
    }
}

/** Convenience: smooth a rotation matrix over time (quaternion-space One Euro). */
class HeadPoseSmoother(minCutoff: Double = 1.0, beta: Double = 0.3) {
    private val q = OneEuroQuaternion(minCutoff, beta)
    fun reset() = q.reset()
    fun filter(R: Mat3, t: Double): Mat3 = quatToRotmat(q.filter(rotmatToQuat(R), t))
}
