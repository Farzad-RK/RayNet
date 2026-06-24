package com.raynet.eyepatch

import kotlin.math.abs

/**
 * JVM harness: runs EyeGeometry through the same canonical ellipses as
 * tools/geom_parity.py and compares against geom_golden.json values.
 */
private const val FOCAL = 1200.0
private const val RADIUS = 40.0

private data class Case(
    val cx: Double, val cy: Double, val w: Double, val h: Double, val rad: Double,
    val ellCo: DoubleArray?,            // expected general conic (null => skip ellipse check)
    val normalPos: DoubleArray?,        // null => expected degenerate (unproject returns null)
    val normalNeg: DoubleArray?,
    val centerPos: DoubleArray?,
    val centerNeg: DoubleArray?,
)

private val CASES = listOf(
    // Case 0: centered + axis-aligned => reference returns NaN => expect null.
    Case(0.0, 0.0, 45.0, 40.0, 0.0,
        doubleArrayOf(1600.0, -0.0, 2025.0, 0.0, 0.0, -3240000.0),
        null, null, null, null),
    Case(20.0, -10.0, 50.0, 35.0, 0.3,
        doubleArrayOf(1336.3485454950799, -719.91915357867, 2388.6514545049195,
            -60653.13335558989, 62171.41216167179, -2145111.605635742),
        doubleArrayOf(-0.2227005266612676, 0.6876927780858891, -0.6910015328444473),
        doubleArrayOf(0.1993964761641394, -0.676027886925029, -0.7093851854901781),
        doubleArrayOf(15.737856710151426, -7.205533187384083, 960.5220671713403),
        doubleArrayOf(16.23017700516249, -8.796132873278001, 960.5006250733549)),
    Case(-30.0, 25.0, 38.0, 30.0, -0.6,
        doubleArrayOf(1073.4386907823448, 507.0292627661712, 1270.5613092176552,
            51730.589877786406, -48317.187577897625, 80323.69289051648),
        doubleArrayOf(0.3263573883501431, 0.5224740167935954, 0.7877256862925428),
        doubleArrayOf(-0.3657975979016836, -0.48960480189925853, 0.7915044253359091),
        doubleArrayOf(-31.914896555319842, 25.80363338287367, 1264.0601094057963),
        doubleArrayOf(-31.223110700519666, 26.815172452332686, 1264.056332681983)),
)

private var failures = 0

private fun cmp(label: String, got: Double, exp: Double, atol: Double, rtol: Double) {
    val tol = atol + rtol * abs(exp)
    val d = abs(got - exp)
    val ok = d <= tol
    if (!ok) failures++
    println("    %-14s got=%+.10g exp=%+.10g  |d|=%.3e  %s".format(label, got, exp, d, if (ok) "ok" else "FAIL"))
}

private fun cmpVec(label: String, got: EyeGeometry.V3, exp: DoubleArray, atol: Double, rtol: Double) {
    cmp("$label.x", got.x, exp[0], atol, rtol)
    cmp("$label.y", got.y, exp[1], atol, rtol)
    cmp("$label.z", got.z, exp[2], atol, rtol)
}

fun main() {
    for ((i, c) in CASES.withIndex()) {
        println("=== case $i: cx=${c.cx} cy=${c.cy} w=${c.w} h=${c.h} rad=${c.rad}")

        // (a) ellipseToGeneral vs golden ell_co
        val ell = EyeGeometry.ellipseToGeneral(c.cx, c.cy, c.w, c.h, c.rad)
        c.ellCo?.let { exp ->
            val names = listOf("A", "B", "C", "D", "E", "F")
            for (k in 0..5) cmp("ell_$k(${names[k]})", ell[k], exp[k], 1e-6, 1e-9)
        }

        // (b) unproject vs golden normals/centers (feed golden ell_co, like the harness)
        val res = EyeGeometry.unproject(FOCAL, c.ellCo ?: ell, RADIUS)
        if (c.normalPos == null) {
            val ok = res == null
            if (!ok) failures++
            println("    expect degenerate(null): got ${if (res == null) "null  ok" else "non-null  FAIL"}")
        } else {
            if (res == null) {
                failures++
                println("    unproject returned null but golden has finite values  FAIL")
            } else {
                cmpVec("normalPos", res.normalPos, c.normalPos!!, 1e-7, 1e-7)
                cmpVec("normalNeg", res.normalNeg, c.normalNeg!!, 1e-7, 1e-7)
                cmpVec("centerPos", res.centerPos, c.centerPos!!, 1e-4, 1e-7)
                cmpVec("centerNeg", res.centerNeg, c.centerNeg!!, 1e-4, 1e-7)
            }
        }
    }
    // (c) reprojection-frame SANITY: unproject's disk center, pushed back
    // through a camera-at-origin pinhole (focal*X/Z), lands NEAR the input
    // ellipse center — confirming EyeModel3D.project()'s frame/scale convention
    // matches unproject's output frame. NOTE: it is NOT exact: the projection of
    // a tilted disk's center differs from the projected-ellipse centroid by
    // perspective foreshortening, and the two-fold (pos/neg) candidates are two
    // differently-tilted disks projecting to the SAME ellipse, so they bracket
    // the true center. So we only assert "within a couple px" (frame sanity).
    println("=== reprojection frame sanity (perspective offset expected, ~<2px)")
    for (c in CASES) {
        if (c.normalPos == null) continue
        val res = EyeGeometry.unproject(FOCAL, c.ellCo!!, RADIUS)!!
        for ((tag, ctr) in listOf("pos" to res.centerPos, "neg" to res.centerNeg)) {
            val px = FOCAL * ctr.x / ctr.z
            val py = FOCAL * ctr.y / ctr.z
            cmp("reproj_$tag.x", px, c.cx, 2.0, 0.0)
            cmp("reproj_$tag.y", py, c.cy, 2.0, 0.0)
        }
    }

    // (d) intersectLines vs Python intersect()
    println("=== intersectLines vs Python intersect()")
    run {
        val a1 = listOf(EyeGeometry.V3(0.0,0.0,0.0), EyeGeometry.V3(10.0,0.0,0.0), EyeGeometry.V3(0.0,10.0,0.0))
        val n1 = listOf(EyeGeometry.V3(0.0,0.0,1.0), EyeGeometry.V3(0.0,1.0,0.0), EyeGeometry.V3(1.0,0.0,0.0))
        cmpVec("int0", EyeGeometry.intersectLines(a1,n1)!!, doubleArrayOf(5.0,5.0,0.0), 1e-9, 1e-9)
        val a2 = listOf(EyeGeometry.V3(1.0,2.0,3.0),EyeGeometry.V3(-4.0,5.0,6.0),EyeGeometry.V3(7.0,-8.0,9.0),EyeGeometry.V3(0.0,0.0,1.0))
        val n2 = listOf(EyeGeometry.V3(1.0,1.0,0.0),EyeGeometry.V3(0.0,1.0,1.0),EyeGeometry.V3(1.0,0.0,1.0),EyeGeometry.V3(1.0,1.0,1.0))
        cmpVec("int1", EyeGeometry.intersectLines(a2,n2)!!,
            doubleArrayOf(-3.80952380952381,-4.52380952380952,-0.666666666666665), 1e-9, 1e-9)
        val a3 = listOf(EyeGeometry.V3(100.0,0.0,500.0),EyeGeometry.V3(-50.0,30.0,480.0),EyeGeometry.V3(20.0,-60.0,520.0))
        val n3 = listOf(EyeGeometry.V3(0.1,0.2,0.97),EyeGeometry.V3(-0.15,0.1,0.98),EyeGeometry.V3(0.05,-0.2,0.978))
        cmpVec("int2", EyeGeometry.intersectLines(a3,n3)!!,
            doubleArrayOf(22.3985528606439,-17.7844242248409,225.887183617963), 1e-6, 1e-9)
    }

    // (e) lineSphereIntersect vs Python line_sphere_intersect() -> [d1(+), d2(-)]
    println("=== lineSphereIntersect vs Python line_sphere_intersect()")
    run {
        fun ls(c: EyeGeometry.V3, r: Double, o: EyeGeometry.V3, l: EyeGeometry.V3) = EyeGeometry.lineSphereIntersect(c,r,o,l)!!
        val d0 = ls(EyeGeometry.V3(0.0,0.0,10.0),5.0,EyeGeometry.V3(0.0,0.0,0.0),EyeGeometry.V3(0.0,0.0,1.0))
        cmp("ls0.d1", d0[0], 15.0, 1e-9, 1e-9); cmp("ls0.d2", d0[1], 5.0, 1e-9, 1e-9)
        val d1 = ls(EyeGeometry.V3(2.0,-3.0,40.0),12.0,EyeGeometry.V3(0.0,0.0,0.0),EyeGeometry.V3(0.1,0.05,1.0))
        cmp("ls1.d1", d1[0], 50.5351362914952, 1e-9, 1e-9); cmp("ls1.d2", d1[1], 29.0688837074973, 1e-9, 1e-9)
        val d2 = ls(EyeGeometry.V3(1.0,1.0,30.0),8.0,EyeGeometry.V3(3.0,-2.0,1.0),EyeGeometry.V3(-0.2,0.1,0.97))
        cmp("ls2.d1", d2[0], 35.9479213200527, 1e-9, 1e-9); cmp("ls2.d2", d2[1], 21.9762359265908, 1e-9, 1e-9)
    }

    // (f) synthetic eyeball-fit round-trip: place a known eyeball (center C,
    // pupil offset d along gaze) and confirm intersectLines + eyeRadius recover
    // it. Exercises the fit math EyeModel3D.finishFit() relies on, with ground truth.
    println("=== synthetic eyeball-fit round-trip (intersectLines + eyeRadius)")
    run {
        val C = EyeGeometry.V3(12.0, -8.0, 980.0)
        val d = 9.7   // pupil-center distance from eyeball center (px)
        // gaze directions on a small cone (camera looks ~ -z toward eye? pupils face camera => -z)
        val dirs = ArrayList<EyeGeometry.V3>(); val pos = ArrayList<EyeGeometry.V3>()
        val angs = listOf(-0.30,-0.20,-0.10,0.0,0.10,0.20,0.30,0.15,-0.15,0.05)
        for ((i,t) in angs.withIndex()) {
            val phi = i * 0.7
            val n = EyeGeometry.V3(kotlin.math.sin(t)*kotlin.math.cos(phi),
                                   kotlin.math.sin(t)*kotlin.math.sin(phi),
                                   -kotlin.math.cos(t)).normalized()  // pupil normal toward camera
            dirs.add(n); pos.add(C + n * d)
        }
        val cFit = EyeGeometry.intersectLines(pos, dirs)!!
        cmpVec("fitCenter", cFit, doubleArrayOf(C.x, C.y, C.z), 1e-6, 1e-9)
        cmp("fitRadius", EyeGeometry.eyeRadius(pos, cFit), d, 1e-6, 1e-9)
    }

    println("\n=== ${if (failures == 0) "ALL PASS" else "$failures CHECK(S) FAILED"} ===")
    if (failures != 0) kotlin.system.exitProcess(1)
}
