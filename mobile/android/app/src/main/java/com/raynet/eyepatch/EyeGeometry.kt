package com.raynet.eyepatch

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.sqrt

/**
 * Classic single-sphere eye-model geometry (the pye3d-free path, Option A of
 * THREEDEEPVOG_DEPLOYMENT.md). Faithful port of 3DeepVOG's
 * `utils/unprojection.py` (Safaee-Rad 1992 conic unprojection) and
 * `utils/intersection.py` (least-squares line intersection + line–sphere).
 *
 * Everything is in DOUBLE to match the numpy reference. Coordinates are in the
 * normalized-patch CAMERA frame: origin at the camera, +z INTO the scene; the
 * camera vertex is `[0, 0, -focal_px]` (image plane at z=0, principal point at
 * the patch center — callers pass center-relative ellipse coords).
 *
 * No refraction correction (per Option A); pupil radius/gaze are a few % biased
 * vs pye3d but fully on-device for both fit and predict.
 */
object EyeGeometry {

    /** A 3-vector as 3 doubles. */
    class V3(val x: Double, val y: Double, val z: Double) {
        operator fun minus(o: V3) = V3(x - o.x, y - o.y, z - o.z)
        operator fun plus(o: V3) = V3(x + o.x, y + o.y, z + o.z)
        operator fun times(s: Double) = V3(x * s, y * s, z * s)
        fun dot(o: V3) = x * o.x + y * o.y + z * o.z
        fun cross(o: V3) = V3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x)
        fun norm() = sqrt(x * x + y * y + z * z)
        fun normalized(): V3 { val n = norm(); return if (n < 1e-12) this else V3(x / n, y / n, z / n) }
    }

    /** One unprojected pupil candidate: a 3D disk center and its gaze normal. */
    class Unprojection(
        val normalPos: V3, val normalNeg: V3,   // camera-frame gaze normals (two-fold)
        val centerPos: V3, val centerNeg: V3,   // camera-frame 3D disk centers
    )

    // ---- Ellipse (center, semi-axes, angle) -> general conic A..F ------------
    // Port of convert_ell_to_general. xc,yc are CENTER-RELATIVE (principal point
    // subtracted) so the cone vertex sits on the optical axis.
    fun ellipseToGeneral(xc: Double, yc: Double, w: Double, h: Double, rad: Double): DoubleArray {
        val s = kotlin.math.sin(rad); val c = cos(rad)
        val A = w * w * s * s + h * h * c * c
        val B = 2.0 * (h * h - w * w) * s * c
        val C = w * w * c * c + h * h * s * s
        val D = -2.0 * A * xc - B * yc
        val E = -B * xc - 2.0 * C * yc
        val F = A * xc * xc + B * xc * yc + C * yc * yc - w * w * h * h
        return doubleArrayOf(A, B, C, D, E, F)
    }

    // ---- Safaee-Rad unprojection --------------------------------------------
    /**
     * @param focalPx camera focal length in pixels (vertex z = -focalPx).
     * @param ellCo   general conic [A,B,C,D,E,F] from [ellipseToGeneral].
     * @param radius  assumed pupil-disk radius (px in the patch scale).
     * @return the two-fold candidate normals + 3D centers, or null if degenerate.
     */
    fun unproject(focalPx: Double, ellCo: DoubleArray, radius: Double): Unprojection? {
        val A = ellCo[0]; val B = ellCo[1]; val C = ellCo[2]
        val D = ellCo[3]; val E = ellCo[4]; val F = ellCo[5]
        val alpha = 0.0; val beta = 0.0; val gamma = -focalPx

        val aP = A; val hP = B / 2.0; val bP = C; val gP = D / 2.0; val fP = E / 2.0; val dP = F
        // gen_cone_co
        val g2 = gamma * gamma
        val a = g2 * aP
        val b = g2 * bP
        val cc = aP * alpha * alpha + 2 * hP * alpha * beta + bP * beta * beta + 2 * gP * alpha + 2 * fP * beta + dP
        val d = g2 * dP
        val f = -gamma * (bP * beta + hP * alpha + fP)
        val g = -gamma * (hP * beta + aP * alpha + gP)
        val h = g2 * hP
        val u = g2 * gP
        val v = g2 * fP
        val w = -gamma * (fP * beta + gP * alpha + dP)

        // Characteristic cubic (Safaee-Rad eq.10): x^3 + c2 x^2 + c1 x + c0.
        val c2 = -(a + b + cc)
        val c1 = b * cc + cc * a + a * b - f * f - g * g - h * h
        val c0 = -(a * b * cc + 2 * f * g * h - a * f * f - b * g * g - cc * h * h)
        val roots = realCubicRootsDesc(c2, c1, c0) ?: return null
        val lamb1 = roots[0]; val lamb2 = roots[1]; val lamb3 = roots[2]

        // gen_lmn (canonical normal, two-fold). With λ1≥λ2≥λ3, l splits ±.
        val lmn = genLmn(lamb1, lamb2, lamb3) ?: return null
        val l0 = lmn[0]; val l1 = lmn[1]; val mm = lmn[2]; val n0 = lmn[3]
        val normCanoPos = doubleArrayOf(l0, mm, n0, 1.0)
        val normCanoNeg = doubleArrayOf(l1, mm, n0, 1.0)

        // T1: rotation of canonical -> camera (columns from gen_rotmat_co).
        val r1 = genRotmat(lamb1, a, b, g, f, h)
        val r2 = genRotmat(lamb2, a, b, g, f, h)
        val r3 = genRotmat(lamb3, a, b, g, f, h)
        // T1 columns are r1,r2,r3 (each is l,m,n of that eigenvector).
        var li = V3(r1[0], r2[0], r3[0])  // row 0 across the 3 eigenvectors
        var mi = V3(r1[1], r2[1], r3[1])  // row 1
        var ni = V3(r1[2], r2[2], r3[2])  // row 2
        if (li.cross(mi).dot(ni) < 0) { li = li * -1.0; mi = mi * -1.0; ni = ni * -1.0 }
        val T1 = doubleArrayOf(
            li.x, li.y, li.z, 0.0,
            mi.x, mi.y, mi.z, 0.0,
            ni.x, ni.y, ni.z, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        val normCamPos = mat4VecRot(T1, normCanoPos)
        val normCamNeg = mat4VecRot(T1, normCanoNeg)

        // T2: translation -(u*li + v*mi + w*ni) / [λ1,λ2,λ3].
        val t2 = doubleArrayOf(
            (u * li.x + v * mi.x + w * ni.x) / lamb1 * -1.0,
            (u * li.y + v * mi.y + w * ni.y) / lamb2 * -1.0,
            (u * li.z + v * mi.z + w * ni.z) / lamb3 * -1.0,
        )
        val T2 = ident4(); T2[3] = t2[0]; T2[7] = t2[1]; T2[11] = t2[2]

        // T3 (pos/neg), ABCD, perfect-frame center.
        val centers = arrayOfNulls<V3>(2)
        val normalsCam = arrayOf(
            V3(normCamPos[0], normCamPos[1], normCamPos[2]),
            V3(normCamNeg[0], normCamNeg[1], normCamNeg[2]),
        )
        val T0 = ident4(); T0[11] = -gamma
        for (k in 0..1) {
            val lk = if (k == 0) l0 else l1
            val T3 = calT3(lk, mm, n0)
            val abcd = calABCD(T3, lamb1, lamb2, lamb3)
            var center = calXYZperfect(abcd[0], abcd[1], abcd[2], abcd[3], radius) ?: return null
            var tc = mat4Vec(T0, mat4Vec(T1, mat4Vec(T2, mat4Vec(T3, center))))
            if (tc[2] < 0) {
                center = doubleArrayOf(-center[0], -center[1], -center[2], center[3])
                tc = mat4Vec(T0, mat4Vec(T1, mat4Vec(T2, mat4Vec(T3, center))))
            }
            centers[k] = V3(tc[0], tc[1], tc[2])
        }
        val result = Unprojection(normalsCam[0], normalsCam[1], centers[0]!!, centers[1]!!)
        // The Safaee-Rad unprojection is DEGENERATE when the conic is symmetric
        // on the optical axis (centered + axis-aligned pupil): gen_rotmat divides
        // by zero and the reference itself returns NaN. Reject such frames so the
        // model never ingests garbage (matches the desktop, which filters NaN).
        if (!result.finite()) return null
        return result
    }

    private fun Unprojection.finite(): Boolean {
        for (v in arrayOf(normalPos, normalNeg, centerPos, centerNeg)) {
            if (!v.x.isFinite() || !v.y.isFinite() || !v.z.isFinite()) return false
        }
        return true
    }

    // ---- Eyeball-center fit: least-squares intersection of gaze lines --------
    /**
     * Port of intersect(a, n): the point minimizing the sum of squared distances
     * to all lines {position + t·normal}. Used over the calibration window to
     * locate the eyeball center. Returns null if the normal system is singular.
     */
    fun intersectLines(positions: List<V3>, normals: List<V3>): V3? {
        // R = I - n n^T (per line); solve (ΣR) p = Σ R a.
        var r00 = 0.0; var r01 = 0.0; var r02 = 0.0
        var r11 = 0.0; var r12 = 0.0; var r22 = 0.0
        var q0 = 0.0; var q1 = 0.0; var q2 = 0.0
        for (i in positions.indices) {
            val n = normals[i].normalized(); val a = positions[i]
            val R00 = 1 - n.x * n.x; val R01 = -n.x * n.y; val R02 = -n.x * n.z
            val R11 = 1 - n.y * n.y; val R12 = -n.y * n.z; val R22 = 1 - n.z * n.z
            r00 += R00; r01 += R01; r02 += R02; r11 += R11; r12 += R12; r22 += R22
            q0 += R00 * a.x + R01 * a.y + R02 * a.z
            q1 += R01 * a.x + R11 * a.y + R12 * a.z
            q2 += R02 * a.x + R12 * a.y + R22 * a.z
        }
        return solveSym3(r00, r01, r02, r11, r12, r22, q0, q1, q2)
    }

    /** Median distance from each line's position to the fitted eyeball center. */
    fun eyeRadius(positions: List<V3>, center: V3): Double {
        val ds = positions.map { (it - center).norm() }.sorted()
        return if (ds.isEmpty()) 0.0 else ds[ds.size / 2]
    }

    // ---- Per-frame: line–sphere intersection (predict) ----------------------
    /**
     * Port of line_sphere_intersect: intersect ray (o + d·l) with the eyeball
     * sphere (center c, radius r). Returns the two parametric distances, or null
     * when the ray misses the sphere.
     */
    fun lineSphereIntersect(c: V3, r: Double, o: V3, l: V3): DoubleArray? {
        val ln = l.normalized()
        val oc = o - c
        val b = ln.dot(oc)
        val delta = b * b - oc.dot(oc) + r * r
        if (delta < 0) return null
        val s = sqrt(delta)
        return doubleArrayOf(-b + s, -b - s)
    }

    // ===================== internal numeric helpers =========================

    /** Real roots of x^3 + c2 x^2 + c1 x + c0, sorted DESCENDING (matches batched_roots). */
    private fun realCubicRootsDesc(c2: Double, c1: Double, c0: Double): DoubleArray? {
        // Depressed cubic t^3 + p t + q via x = t - c2/3.
        val p = c1 - c2 * c2 / 3.0
        val q = 2.0 * c2 * c2 * c2 / 27.0 - c2 * c1 / 3.0 + c0
        val shift = c2 / 3.0
        val out: DoubleArray
        if (abs(p) < 1e-12) {
            // Triple/degenerate root.
            val t = Math.cbrt(-q)
            out = doubleArrayOf(t - shift, t - shift, t - shift)
        } else {
            val m = 2.0 * sqrt(-p / 3.0)
            var arg = (3.0 * q) / (p * m)   // = (3q)/(2p) * sqrt(-3/p)
            if (arg < -1.0) arg = -1.0; if (arg > 1.0) arg = 1.0
            val theta = acos(arg) / 3.0
            val twoPi3 = 2.0 * Math.PI / 3.0
            val r0 = m * cos(theta) - shift
            val r1 = m * cos(theta - twoPi3) - shift
            val r2 = m * cos(theta - 2.0 * twoPi3) - shift
            out = doubleArrayOf(r0, r1, r2)
        }
        out.sort(); // ascending
        return doubleArrayOf(out[2], out[1], out[0]) // descending
    }

    /** gen_lmn with λ1≥λ2≥λ3 → returns [l_pos, l_neg, m, n]. */
    private fun genLmn(l1: Double, l2: Double, l3: Double): DoubleArray? {
        return when {
            l1 > l2 -> {
                val lp = sqrt((l1 - l2) / (l1 - l3))
                val n = sqrt((l2 - l3) / (l1 - l3))
                doubleArrayOf(lp, -lp, 0.0, n)
            }
            l1 < l2 -> {
                // m splits instead of l (kept for completeness; rare with desc sort).
                val mp = sqrt((l2 - l1) / (l2 - l3))
                val n = sqrt((l1 - l3) / (l2 - l3))
                doubleArrayOf(0.0, 0.0, mp, n) // note: m two-fold handled as ±mp upstream if needed
            }
            else -> doubleArrayOf(0.0, 0.0, 0.0, 1.0)
        }
    }

    /** gen_rotmat_co: eigenvector (l,m,n) for eigenvalue `lamb`. */
    private fun genRotmat(lamb: Double, a: Double, b: Double, g: Double, f: Double, h: Double): DoubleArray {
        val t1 = (b - lamb) * g - f * h
        val t2 = (a - lamb) * f - g * h
        val t3 = -(a - lamb) * (t1 / t2) / g - (h / g)
        val m = 1.0 / sqrt(1.0 + (t1 / t2) * (t1 / t2) + t3 * t3)
        val l = (t1 / t2) * m
        val n = t3 * m
        return doubleArrayOf(l, m, n)
    }

    private fun calT3(l: Double, m: Double, n: Double): DoubleArray {
        val lm = sqrt(l * l + m * m)
        return doubleArrayOf(
            -m / lm, -(l * n) / lm, l, 0.0,
            l / lm, -(m * n) / lm, m, 0.0,
            0.0, lm, n, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    private fun calABCD(T3: DoubleArray, l1: Double, l2: Double, l3: Double): DoubleArray {
        // Columns 0,1,2 (rows 0..2) of T3 are li, mi, ni.
        val li = doubleArrayOf(T3[0], T3[4], T3[8])
        val ni = doubleArrayOf(T3[2], T3[6], T3[10])
        val lam = doubleArrayOf(l1, l2, l3)
        var A = 0.0; var Bb = 0.0; var Cc = 0.0; var Dd = 0.0
        val mi = doubleArrayOf(T3[1], T3[5], T3[9])
        for (i in 0..2) {
            A += li[i] * li[i] * lam[i]
            Bb += li[i] * ni[i] * lam[i]
            Cc += mi[i] * ni[i] * lam[i]
            Dd += ni[i] * ni[i] * lam[i]
        }
        return doubleArrayOf(A, Bb, Cc, Dd)
    }

    private fun calXYZperfect(A: Double, B: Double, C: Double, D: Double, r: Double): DoubleArray? {
        val disc = B * B + C * C - A * D
        if (disc <= 0) return null
        val Z = (A * r) / sqrt(disc)
        val X = (-B / A) * Z
        val Y = (-C / A) * Z
        return doubleArrayOf(X, Y, Z, 1.0)
    }

    private fun ident4() = doubleArrayOf(
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    )

    /** 4x4 (row-major) times homogeneous 4-vector. */
    private fun mat4Vec(M: DoubleArray, v: DoubleArray): DoubleArray {
        val o = DoubleArray(4)
        for (row in 0..3) {
            o[row] = M[row * 4] * v[0] + M[row * 4 + 1] * v[1] +
                     M[row * 4 + 2] * v[2] + M[row * 4 + 3] * v[3]
        }
        return o
    }

    /** Rotation-only 4x4 applied to a direction (ignores translation/w). */
    private fun mat4VecRot(M: DoubleArray, v: DoubleArray) = mat4Vec(M, v)

    /** Solve a symmetric 3x3 system [r·] p = q via Cramer; null if near-singular. */
    private fun solveSym3(
        r00: Double, r01: Double, r02: Double, r11: Double, r12: Double, r22: Double,
        q0: Double, q1: Double, q2: Double,
    ): V3? {
        val det = r00 * (r11 * r22 - r12 * r12) -
                  r01 * (r01 * r22 - r12 * r02) +
                  r02 * (r01 * r12 - r11 * r02)
        if (abs(det) < 1e-12) return null
        val inv = 1.0 / det
        val i00 = (r11 * r22 - r12 * r12) * inv
        val i01 = (r02 * r12 - r01 * r22) * inv
        val i02 = (r01 * r12 - r02 * r11) * inv
        val i11 = (r00 * r22 - r02 * r02) * inv
        val i12 = (r02 * r01 - r00 * r12) * inv
        val i22 = (r00 * r11 - r01 * r01) * inv
        return V3(
            i00 * q0 + i01 * q1 + i02 * q2,
            i01 * q0 + i11 * q1 + i12 * q2,
            i02 * q0 + i12 * q1 + i22 * q2,
        )
    }
}
