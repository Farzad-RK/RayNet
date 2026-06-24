package com.raynet.eyepatch

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.hypot
import kotlin.random.Random

/**
 * A gaze-driven mini-game demoing the tracker: a toolbar of four shapes sits along
 * the top; a ring in the centre asks for a randomly chosen shape. The user picks a
 * shape up by DWELLING their gaze on it, carries it on the cursor, and drops it by
 * dwelling on the ring. A correct match scores and re-rolls the request; a wrong
 * one flashes and is returned.
 *
 * Driven by [onCursor] (the calibrated normalized gaze point from
 * [CalibrationView.mapToScreen]); selection uses the same dwell idea as calibration
 * since the cursor updates at only a few Hz.
 */
class GameView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    enum class Shape { TRIANGLE, SQUARE, RECTANGLE, CIRCLE }

    var onStatus: ((String) -> Unit)? = null

    private val shapes = Shape.values()
    private var required = shapes[0]
    private var held: Shape? = null
    private var score = 0

    private var cursor: DoubleArray? = null          // smoothed normalized cursor
    private var dwellTarget = -1                      // 0..3 toolbar, 4 ring, -1 none
    private var dwellStartT = Double.NaN
    private var flashT = Double.NaN                   // brief correct/wrong feedback
    private var flashOk = false
    private val dwellS = 0.8
    private val flashS = 0.7
    private val cursorAlpha = 0.5

    private val rng = Random(System.nanoTime())

    private val bg = Color.WHITE
    private val toolbarBg = Color.rgb(244, 246, 250)
    private val shapeFill = fill(Color.rgb(70, 110, 200))
    private val ringStroke = stroke(Color.rgb(120, 120, 128), 8f)
    private val reqHint = Paint().apply { color = Color.rgb(200, 205, 215); style = Paint.Style.FILL; isAntiAlias = true }
    private val dwellArc = stroke(Color.rgb(255, 170, 0), 9f)
    private val cursorPaint = fill(Color.rgb(30, 110, 220))
    private val cursorRing = stroke(Color.rgb(30, 110, 220), 3f)
    private val okPaint = fill(Color.argb(60, 40, 180, 80))
    private val wrongPaint = fill(Color.argb(60, 210, 50, 50))
    private val text = Paint().apply { color = Color.rgb(40, 40, 40); textSize = 40f; isAntiAlias = true }
    private val textC = Paint().apply {
        color = Color.rgb(40, 40, 40); textSize = 34f; isAntiAlias = true; textAlign = Paint.Align.CENTER
    }

    private fun fill(c: Int) = Paint().apply { color = c; style = Paint.Style.FILL; isAntiAlias = true }
    private fun stroke(c: Int, w: Float) =
        Paint().apply { color = c; style = Paint.Style.STROKE; strokeWidth = w; isAntiAlias = true }

    init { reroll() }

    private fun reroll() { required = shapes[rng.nextInt(shapes.size)] }

    /** One frame: @param p calibrated normalized gaze point (null if uncalibrated/lost). */
    fun onCursor(p: DoubleArray?, t: Double) {
        lastT = t
        if (!flashT.isNaN() && t - flashT > flashS) flashT = Double.NaN
        if (p == null) {
            cursor = null; dwellTarget = -1; dwellStartT = Double.NaN
            onStatus?.invoke("Calibrate on the Calibration tab first, then look at a shape.")
            invalidate(); return
        }
        // Smooth the cursor for steadier dwell.
        val c = cursor
        cursor = if (c == null) p
                 else doubleArrayOf(c[0] + cursorAlpha * (p[0] - c[0]), c[1] + cursorAlpha * (p[1] - c[1]))
        updateDwell(t)
        invalidate()
    }

    private fun updateDwell(t: Double) {
        val cur = cursor ?: return
        val tgt = hitTest(cur)
        if (tgt != dwellTarget) { dwellTarget = tgt; dwellStartT = if (tgt >= 0) t else Double.NaN }
        if (tgt < 0) { onStatus?.invoke(prompt()); return }
        if (dwellStartT.isNaN()) dwellStartT = t
        if (t - dwellStartT >= dwellS) { trigger(tgt, t); dwellStartT = t }
        else onStatus?.invoke(prompt())
    }

    private fun trigger(tgt: Int, t: Double) {
        if (tgt < 4) {                          // toolbar slot -> pick up
            held = shapes[tgt]
        } else {                                // ring -> drop
            val h = held ?: return
            if (h == required) { score++; flashOk = true; reroll() } else { flashOk = false }
            held = null
            flashT = t
        }
    }

    private fun prompt(): String {
        val base = "Score $score  •  Place: ${required.name.lowercase()}"
        return if (held != null) "$base  •  holding ${held!!.name.lowercase()} — look at the ring"
               else "$base  •  look at the matching shape to pick it up"
    }

    // 0..3 = toolbar slots, 4 = centre ring, -1 = none.
    private fun hitTest(p: DoubleArray): Int {
        val x = p[0] * width; val y = p[1] * height
        val toolbarH = height * 0.18f
        if (y <= toolbarH) {
            val slot = (p[0] * 4).toInt().coerceIn(0, 3)
            return slot
        }
        val cx = width / 2f; val cy = height * 0.58f
        val ringR = minOf(width, height) * 0.18f
        if (hypot((x - cx).toDouble(), (y - cy).toDouble()) <= ringR * 1.15) return 4
        return -1
    }

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(bg)
        val w = width.toFloat(); val h = height.toFloat()
        val toolbarH = h * 0.18f
        canvas.drawRect(0f, 0f, w, toolbarH, fill(toolbarBg))

        // Toolbar shapes.
        val slotW = w / 4f
        val icon = toolbarH * 0.36f
        for (i in shapes.indices) {
            val cx = slotW * (i + 0.5f); val cy = toolbarH * 0.5f
            if (held != shapes[i]) drawShape(canvas, shapes[i], cx, cy, icon, shapeFill)
            if (dwellTarget == i) drawDwellArc(canvas, cx, cy, icon * 1.5f)
        }

        // Centre ring + the requested shape as a faint hint inside it.
        val cx = w / 2f; val cy = h * 0.58f
        val ringR = minOf(w, h) * 0.18f
        canvas.drawCircle(cx, cy, ringR, ringStroke)
        drawShape(canvas, required, cx, cy, ringR * 0.5f, reqHint)
        if (dwellTarget == 4) drawDwellArc(canvas, cx, cy, ringR * 1.25f)

        // Correct/wrong flash over the ring.
        if (!flashT.isNaN()) canvas.drawCircle(cx, cy, ringR, if (flashOk) okPaint else wrongPaint)

        // Held shape rides the cursor; cursor dot.
        cursor?.let {
            val px = it[0].toFloat() * w; val py = it[1].toFloat() * h
            held?.let { hs -> drawShape(canvas, hs, px, py, toolbarH * 0.30f, shapeFill) }
            canvas.drawCircle(px, py, 14f, cursorPaint)
            canvas.drawCircle(px, py, 26f, cursorRing)
        }

        canvas.drawText("Score: $score", 24f, h - 28f, text)
        canvas.drawText("Place the ${required.name.lowercase()} in the ring", w / 2, h - 28f, textC)
    }

    private fun drawDwellArc(c: Canvas, cx: Float, cy: Float, r: Float) {
        val frac = if (dwellStartT.isNaN()) 0f else ((lastT - dwellStartT) / dwellS).toFloat().coerceIn(0f, 1f)
        c.drawArc(RectF(cx - r, cy - r, cx + r, cy + r), -90f, 360f * frac, false, dwellArc)
    }

    private var lastT = 0.0

    private fun drawShape(c: Canvas, s: Shape, cx: Float, cy: Float, r: Float, p: Paint) {
        when (s) {
            Shape.CIRCLE -> c.drawCircle(cx, cy, r, p)
            Shape.SQUARE -> c.drawRect(cx - r, cy - r, cx + r, cy + r, p)
            Shape.RECTANGLE -> c.drawRect(cx - r * 1.4f, cy - r * 0.75f, cx + r * 1.4f, cy + r * 0.75f, p)
            Shape.TRIANGLE -> {
                val path = Path().apply {
                    moveTo(cx, cy - r); lineTo(cx - r, cy + r); lineTo(cx + r, cy + r); close()
                }
                c.drawPath(path, p)
            }
        }
    }

}
