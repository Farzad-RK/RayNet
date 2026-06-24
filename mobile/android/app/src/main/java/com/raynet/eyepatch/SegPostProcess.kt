package com.raynet.eyepatch

/**
 * Tier-2 post-processing of the segmentation probability planes — port of
 * `threedeepvog/module/PostProcessing.py` (the "morph" + "largest" cleanup paths).
 *
 * For each requested channel: threshold at its probability, then
 *   morphological OPEN (erosion -> dilation, 3x3, `morphIters` times) to drop
 *   speckle, then keep only the LARGEST 4-connected component (the actual
 *   pupil / iris blob, dropping stray reflections). The result is a clean
 *   boolean mask per channel, plus the original probabilities kept for the
 *   confidence score downstream.
 *
 * Pure logic (no Android types) so it can be unit-tested on the JVM.
 */
object SegPostProcess {

    /** Cleaned boolean masks (row-major, length w*h) + the source probabilities. */
    class Clean(
        val pupilMask: BooleanArray, val irisMask: BooleanArray, val scleraMask: BooleanArray,
        val pupilProb: FloatArray, val irisProb: FloatArray,
        val w: Int, val h: Int,
    )

    fun process(
        seg: SegMaps,
        thPupil: Double = 0.5, thIris: Double = 0.5, thSclera: Double = 0.5,
        morphIters: Int = 0, largestComponent: Boolean = true,
    ): Clean {
        val w = seg.w; val h = seg.h
        val pupil = cleanChannel(seg.pupil, w, h, thPupil, morphIters, largestComponent)
        val iris = cleanChannel(seg.iris, w, h, thIris, morphIters, largestComponent)
        // Sclera is only used as a context mask (blink + iris masking); threshold only.
        val sclera = threshold(seg.sclera, thSclera)
        return Clean(pupil, iris, sclera, seg.pupil, seg.iris, w, h)
    }

    private fun cleanChannel(
        prob: FloatArray, w: Int, h: Int, th: Double, iters: Int, largest: Boolean,
    ): BooleanArray {
        var m = threshold(prob, th)
        repeat(iters) { m = dilate(erode(m, w, h), w, h) }   // morphological opening
        if (largest) m = largestComponent(m, w, h)
        return m
    }

    fun threshold(prob: FloatArray, th: Double): BooleanArray {
        val t = th.toFloat()
        return BooleanArray(prob.size) { prob[it] > t }
    }

    // 3x3 erosion: a pixel survives only if all 8 neighbours (and itself) are set.
    // Out-of-bounds neighbours count as 0 (matches kornia's zero-padded erosion).
    private fun erode(m: BooleanArray, w: Int, h: Int): BooleanArray {
        val out = BooleanArray(m.size)
        for (y in 0 until h) for (x in 0 until w) {
            val i = y * w + x
            if (!m[i]) continue
            var keep = true
            loop@ for (dy in -1..1) for (dx in -1..1) {
                val nx = x + dx; val ny = y + dy
                if (nx < 0 || ny < 0 || nx >= w || ny >= h || !m[ny * w + nx]) { keep = false; break@loop }
            }
            out[i] = keep
        }
        return out
    }

    // 3x3 dilation: a pixel is set if any 8-neighbour (or itself) is set.
    private fun dilate(m: BooleanArray, w: Int, h: Int): BooleanArray {
        val out = BooleanArray(m.size)
        for (y in 0 until h) for (x in 0 until w) {
            val i = y * w + x
            if (!m[i]) continue
            for (dy in -1..1) for (dx in -1..1) {
                val nx = x + dx; val ny = y + dy
                if (nx in 0 until w && ny in 0 until h) out[ny * w + nx] = true
            }
        }
        return out
    }

    /** Keep only the largest 4-connected component (iterative flood fill). */
    fun largestComponent(m: BooleanArray, w: Int, h: Int): BooleanArray {
        val label = IntArray(m.size) { -1 }
        val stack = IntArray(m.size)
        var best = -1; var bestSize = 0
        var cur = 0
        for (start in m.indices) {
            if (!m[start] || label[start] != -1) continue
            var sp = 0; stack[sp++] = start; label[start] = cur
            var size = 0
            while (sp > 0) {
                val p = stack[--sp]; size++
                val x = p % w; val y = p / w
                if (x > 0)     { val q = p - 1; if (m[q] && label[q] == -1) { label[q] = cur; stack[sp++] = q } }
                if (x < w - 1) { val q = p + 1; if (m[q] && label[q] == -1) { label[q] = cur; stack[sp++] = q } }
                if (y > 0)     { val q = p - w; if (m[q] && label[q] == -1) { label[q] = cur; stack[sp++] = q } }
                if (y < h - 1) { val q = p + w; if (m[q] && label[q] == -1) { label[q] = cur; stack[sp++] = q } }
            }
            if (size > bestSize) { bestSize = size; best = cur }
            cur++
        }
        if (best < 0) return BooleanArray(m.size)
        return BooleanArray(m.size) { label[it] == best }
    }
}
