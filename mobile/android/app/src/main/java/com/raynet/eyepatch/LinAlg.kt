package com.raynet.eyepatch

/**
 * Minimal 3x3 / 3-vector linear algebra, in Double to match the NumPy (float64)
 * reference in eye_norm.py / filters.py. Kept tiny and dependency-free on purpose
 * — this is all the math the on-device warp + quaternion smoothing need, so we
 * don't pull in OpenCV just for a handful of operations.
 */

/** A 3-vector. */
data class Vec3(val x: Double, val y: Double, val z: Double) {
    operator fun plus(o: Vec3) = Vec3(x + o.x, y + o.y, z + o.z)
    operator fun minus(o: Vec3) = Vec3(x - o.x, y - o.y, z - o.z)
    operator fun times(s: Double) = Vec3(x * s, y * s, z * s)
    fun dot(o: Vec3) = x * o.x + y * o.y + z * o.z
    fun cross(o: Vec3) = Vec3(
        y * o.z - z * o.y,
        z * o.x - x * o.z,
        x * o.y - y * o.x,
    )
    fun norm() = kotlin.math.sqrt(dot(this))
    fun normalized(): Vec3 {
        val n = norm()
        return if (n < 1e-12) this else this * (1.0 / n)
    }
}

/**
 * A 3x3 matrix stored row-major in a DoubleArray(9):
 *   [ m[0] m[1] m[2] ]
 *   [ m[3] m[4] m[5] ]
 *   [ m[6] m[7] m[8] ]
 */
class Mat3(val m: DoubleArray) {
    init { require(m.size == 9) { "Mat3 needs 9 elements" } }

    operator fun get(r: Int, c: Int) = m[r * 3 + c]

    /** Column c as a Vec3. */
    fun col(c: Int) = Vec3(m[c], m[3 + c], m[6 + c])

    operator fun times(o: Mat3): Mat3 {
        val r = DoubleArray(9)
        for (i in 0..2) for (j in 0..2) {
            var s = 0.0
            for (k in 0..2) s += this[i, k] * o[k, j]
            r[i * 3 + j] = s
        }
        return Mat3(r)
    }

    fun mul(v: Vec3) = Vec3(
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[3] * v.x + m[4] * v.y + m[5] * v.z,
        m[6] * v.x + m[7] * v.y + m[8] * v.z,
    )

    fun inverse(): Mat3 {
        val a = m
        val det =
            a[0] * (a[4] * a[8] - a[5] * a[7]) -
            a[1] * (a[3] * a[8] - a[5] * a[6]) +
            a[2] * (a[3] * a[7] - a[4] * a[6])
        require(kotlin.math.abs(det) > 1e-15) { "singular Mat3" }
        val inv = 1.0 / det
        return Mat3(doubleArrayOf(
            (a[4] * a[8] - a[5] * a[7]) * inv,
            (a[2] * a[7] - a[1] * a[8]) * inv,
            (a[1] * a[5] - a[2] * a[4]) * inv,
            (a[5] * a[6] - a[3] * a[8]) * inv,
            (a[0] * a[8] - a[2] * a[6]) * inv,
            (a[2] * a[3] - a[0] * a[5]) * inv,
            (a[3] * a[7] - a[4] * a[6]) * inv,
            (a[1] * a[6] - a[0] * a[7]) * inv,
            (a[0] * a[4] - a[1] * a[3]) * inv,
        ))
    }

    /** Transpose (for a rotation matrix, the inverse). */
    fun transpose() = Mat3(doubleArrayOf(
        m[0], m[3], m[6],
        m[1], m[4], m[7],
        m[2], m[5], m[8],
    ))

    /** Row-major float[9] for android.graphics.Matrix.setValues(...). */
    fun toFloat9(): FloatArray = FloatArray(9) { m[it].toFloat() }

    companion object {
        /** Build from three ROW vectors (rows r0,r1,r2). */
        fun fromRows(r0: Vec3, r1: Vec3, r2: Vec3) = Mat3(doubleArrayOf(
            r0.x, r0.y, r0.z,
            r1.x, r1.y, r1.z,
            r2.x, r2.y, r2.z,
        ))
    }
}
