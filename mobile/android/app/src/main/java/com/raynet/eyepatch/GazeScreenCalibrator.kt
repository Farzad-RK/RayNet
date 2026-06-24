package com.raynet.eyepatch

import com.raynet.eyepatch.EyeGeometry.V3
import kotlin.math.atan2

/**
 * Per-user 2D->2D mapping fit, the LEARNED half of the hybrid gaze->screen model.
 *
 * Input is a 2-vector `(u,v)` of features and output the normalized screen point
 * `(nx,ny)`. In hybrid mode `(u,v)` is the geometric coarse point from
 * [ScreenGeometry]; the fitted polynomial is then a small RESIDUAL that removes the
 * systematic errors geometry cannot know from a fixed pose — the per-person kappa
 * angle, the focal/scale guess, axis sign, and pupil foreshortening. In the
 * geometry-unavailable fallback `(u,v)` is the raw gaze `(yaw,pitch)` (see
 * [angles]) and the polynomial is the whole mapping, as before.
 *
 * A full quadratic (6 terms) absorbs mild barrel distortion; with too few targets
 * it degrades to affine (3 terms). A tiny ridge keeps the normal equations stable.
 */
class GazeScreenCalibrator {

    private data class Obs(val nx: Double, val ny: Double, val u: Double, val v: Double)

    private val obs = ArrayList<Obs>()
    private var cX: DoubleArray? = null   // screen-x coefficients
    private var cY: DoubleArray? = null   // screen-y coefficients
    private var nFeat = 0

    val isCalibrated get() = cX != null
    val targetCount get() = obs.size

    fun reset() { obs.clear(); cX = null; cY = null; nFeat = 0 }

    /** Record one fixated target: its normalized position + the (u,v) samples held. */
    fun addTarget(nx: Double, ny: Double, samples: List<DoubleArray>) {
        if (samples.isEmpty()) return
        val us = samples.map { it[0] }.sorted()
        val vs = samples.map { it[1] }.sorted()
        obs.add(Obs(nx, ny, median(us), median(vs)))
    }

    /** Fit the mapping. Quadratic with >=6 targets, else affine with >=3. */
    fun fit(): Boolean {
        nFeat = when {
            obs.size >= 6 -> 6
            obs.size >= 3 -> 3
            else -> return false
        }
        val rows = obs.map { feat(it.u, it.v) }
        val bx = obs.map { it.nx }.toDoubleArray()
        val by = obs.map { it.ny }.toDoubleArray()
        val x = solveLeastSquares(rows, bx) ?: return false
        val y = solveLeastSquares(rows, by) ?: return false
        cX = x; cY = y
        return true
    }

    /** Predict the normalized screen point for a feature pair, or null if unfit. */
    fun predict(u: Double, v: Double): DoubleArray? {
        val ax = cX ?: return null
        val ay = cY ?: return null
        val f = feat(u, v)
        var nx = 0.0; var ny = 0.0
        for (i in 0 until nFeat) { nx += ax[i] * f[i]; ny += ay[i] * f[i] }
        return doubleArrayOf(nx.coerceIn(0.0, 1.0), ny.coerceIn(0.0, 1.0))
    }

    fun predict(uv: DoubleArray): DoubleArray? = predict(uv[0], uv[1])

    // Quadratic feature vector (first nFeat used): [1, u, v, u^2, v^2, u*v].
    private fun feat(u: Double, v: Double) = doubleArrayOf(1.0, u, v, u * u, v * v, u * v)

    private fun median(s: List<Double>) = s[s.size / 2]

    companion object {
        /** Gaze direction (camera frame, looking back at camera ~ (0,0,-1)) -> [yaw,pitch]. */
        fun angles(g: V3): DoubleArray {
            val fwd = if (-g.z > 1e-6) -g.z else 1e-6   // depth toward the viewer
            return doubleArrayOf(atan2(g.x, fwd), atan2(g.y, fwd))
        }
    }

    /**
     * Solve min_c ||A c - b||^2 via the normal equations (A^T A) c = A^T b, with
     * a tiny ridge term for conditioning. A is rows x nFeat; returns the nFeat
     * coefficients, or null if singular.
     */
    private fun solveLeastSquares(rows: List<DoubleArray>, b: DoubleArray): DoubleArray? {
        val n = nFeat
        val ata = Array(n) { DoubleArray(n) }
        val atb = DoubleArray(n)
        for (r in rows.indices) {
            val a = rows[r]
            for (i in 0 until n) {
                atb[i] += a[i] * b[r]
                for (j in 0 until n) ata[i][j] += a[i] * a[j]
            }
        }
        for (i in 0 until n) ata[i][i] += 1e-6   // ridge
        return gaussSolve(ata, atb)
    }

    /** Gaussian elimination with partial pivoting; returns null if singular. */
    private fun gaussSolve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray? {
        val n = b.size
        val m = Array(n) { i -> DoubleArray(n + 1).also { row ->
            for (j in 0 until n) row[j] = a[i][j]; row[n] = b[i]
        } }
        for (col in 0 until n) {
            var piv = col
            for (r in col + 1 until n) if (kotlin.math.abs(m[r][col]) > kotlin.math.abs(m[piv][col])) piv = r
            if (kotlin.math.abs(m[piv][col]) < 1e-12) return null
            val tmp = m[col]; m[col] = m[piv]; m[piv] = tmp
            for (r in 0 until n) {
                if (r == col) continue
                val f = m[r][col] / m[col][col]
                for (k in col..n) m[r][k] -= f * m[col][k]
            }
        }
        return DoubleArray(n) { i -> m[i][n] / m[i][i] }
    }
}
