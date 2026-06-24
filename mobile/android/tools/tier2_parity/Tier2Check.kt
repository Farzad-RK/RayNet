package com.raynet.eyepatch

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/** JVM check: synthesize known filled ellipses, confirm EllipseFitter recovers
 *  (cx,cy,w,h,radian), and that SegPostProcess.largestComponent drops speckle. */
private var failures = 0
private fun cmp(label: String, got: Double, exp: Double, tol: Double) {
    val ok = abs(got - exp) <= tol
    if (!ok) failures++
    println("    %-12s got=%.4f exp=%.4f |d|=%.4f %s".format(label, got, exp, abs(got - exp), if (ok) "ok" else "FAIL"))
}

private fun renderEllipse(w: Int, h: Int, cx: Double, cy: Double, a: Double, b: Double, th: Double): BooleanArray {
    val m = BooleanArray(w * h)
    val ct = cos(th); val st = sin(th)
    for (y in 0 until h) for (x in 0 until w) {
        val dx = x - cx; val dy = y - cy
        val xr = dx * ct + dy * st; val yr = -dx * st + dy * ct
        if ((xr / a) * (xr / a) + (yr / b) * (yr / b) <= 1.0) m[y * w + x] = true
    }
    return m
}

fun main() {
    val w = 320; val h = 240
    data class C(val cx: Double, val cy: Double, val a: Double, val b: Double, val th: Double)
    val cases = listOf(
        C(160.0, 120.0, 40.0, 25.0, 0.0),
        C(140.0, 100.0, 50.0, 30.0, 0.6),
        C(180.0, 130.0, 35.0, 20.0, -0.9),
    )
    for ((i, c) in cases.withIndex()) {
        println("=== ellipse $i: c=(${c.cx},${c.cy}) a=${c.a} b=${c.b} th=${c.th}")
        val mask = renderEllipse(w, h, c.cx, c.cy, c.a, c.b, c.th)
        val prob = FloatArray(mask.size) { if (mask[it]) 1f else 0f }
        val f = EllipseFitter.fit(mask, prob, w, h)
        if (!f.valid) { failures++; println("    FAIL: invalid fit"); continue }
        cmp("cx", f.cx, c.cx, 1.0); cmp("cy", f.cy, c.cy, 1.0)
        cmp("w(semiMaj)", f.w, c.a, 1.5); cmp("h(semiMin)", f.h, c.b, 1.5)
        // angle: compare as direction (mod pi), allow wrap
        var dth = abs(f.radian - c.th) % Math.PI
        if (dth > Math.PI / 2) dth = Math.PI - dth
        cmp("radian", dth, 0.0, 0.05)
        cmp("confidence", f.confidence, 1.0, 0.05)
    }

    // largestComponent: big blob + a far speckle -> speckle removed.
    println("=== largestComponent drops speckle")
    val m = renderEllipse(w, h, 160.0, 120.0, 30.0, 20.0, 0.0)
    m[5 * w + 5] = true; m[5 * w + 6] = true   // tiny speckle in the corner
    val cleaned = SegPostProcess.largestComponent(m, w, h)
    val speckleGone = !cleaned[5 * w + 5] && !cleaned[5 * w + 6]
    val blobKept = cleaned[120 * w + 160]
    if (!speckleGone || !blobKept) failures++
    println("    speckleGone=$speckleGone blobKept=$blobKept ${if (speckleGone && blobKept) "ok" else "FAIL"}")

    println("\n=== ${if (failures == 0) "ALL PASS" else "$failures CHECK(S) FAILED"} ===")
    if (failures != 0) kotlin.system.exitProcess(1)
}
