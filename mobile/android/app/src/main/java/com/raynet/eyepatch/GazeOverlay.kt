package com.raynet.eyepatch

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import kotlin.math.hypot

/**
 * Tier-2/3 annotation rendering — port of `ParamsRender.overlay_fit_model_cv` +
 * `realtime_gaze.draw_gaze`. Draws onto a COPY of the 320x240 eye patch with
 * `Canvas`/`Paint` (no OpenCV): pupil + iris ellipses, segmentation tint, gaze
 * arrow, eyeball + pupil centers, and the pupil-radius / gaze-angle text.
 *
 * The gaze arrow uses the lateral (x,y) components of the 3D gaze normal, so it
 * is zero looking straight ahead and grows as the eye turns.
 */
object GazeOverlay {

    private const val ARROW_PX = 130.0   // arrow length at |lateral gaze| = 1

    private val pupilPaint = stroke(Color.GREEN, 2f)
    private val irisPaint = stroke(Color.MAGENTA, 2f)
    private val gazePaint = stroke(Color.YELLOW, 3f)
    private val eyeCenterPaint = fill(Color.CYAN)
    private val pupilCenterPaint = fill(Color.GREEN)
    private val pupilTintPaint = Paint().apply { color = Color.argb(70, 0, 255, 0) }
    private val irisTintPaint = Paint().apply { color = Color.argb(45, 255, 0, 255) }
    private val textPaint = Paint().apply {
        color = Color.WHITE; textSize = 18f; isAntiAlias = true
        setShadowLayer(2f, 1f, 1f, Color.BLACK)
    }
    private val warnPaint = Paint().apply {
        color = Color.rgb(255, 140, 0); textSize = 20f; isAntiAlias = true
        setShadowLayer(2f, 1f, 1f, Color.BLACK)
    }

    private fun stroke(c: Int, wpx: Float) = Paint().apply {
        color = c; style = Paint.Style.STROKE; strokeWidth = wpx; isAntiAlias = true
    }
    private fun fill(c: Int) = Paint().apply { color = c; style = Paint.Style.FILL; isAntiAlias = true }

    /** @return a new annotated bitmap (input patch is not modified). */
    fun render(patch: Bitmap, r: EyeTracker.Result, showSegTint: Boolean = true): Bitmap {
        val out = patch.copy(Bitmap.Config.ARGB_8888, true)
        val c = Canvas(out)

        r.clean?.let { if (showSegTint) drawSegTint(c, it) }
        r.iris?.let { drawEllipse(c, it, irisPaint) }
        r.pupil?.let { drawEllipse(c, it, pupilPaint); c.drawCircle(it.cx.toFloat(), it.cy.toFloat(), 2.5f, pupilCenterPaint) }

        val gaze = r.gaze
        val pupil = r.pupil
        if (gaze != null && pupil != null) {
            // Anchor the arrow at the DIRECTLY-MEASURED 2D pupil center (reliable)
            // and use the 3D gaze normal only for DIRECTION. The tier-3 model's
            // 3D translation (projected pupil/eyeball center) is reconstructed and
            // not yet validated against desktop 3DeepVOG, so we do NOT trust its
            // absolute projected position or its mm pupil radius here.
            val px = pupil.cx.toFloat(); val py = pupil.cy.toFloat()
            val tx = (px + ARROW_PX * gaze.gaze.x).toFloat()
            val ty = (py + ARROW_PX * gaze.gaze.y).toFloat()
            drawArrow(c, px, py, tx, ty)
            // Apparent (2D, foreshortened) pupil diameter — directly measured.
            c.drawText("Øap=%.0fpx".format(2 * pupil.w), 6f, patch.height - 10f, textPaint)
        }

        // Status corner text.
        val statusY = 20f
        when (r.mode) {
            EyeTracker.Mode.CALIBRATING ->
                c.drawText("calib ${r.calibCount}", 6f, statusY, warnPaint)
            EyeTracker.Mode.TRACKING ->
                if (gaze == null) c.drawText("no pupil", 6f, statusY, warnPaint)
            EyeTracker.Mode.IDLE ->
                if (r.pupil == null) c.drawText("no pupil", 6f, statusY, warnPaint)
        }
        if (r.blink) c.drawText("blink?", patch.width - 60f, statusY, warnPaint)
        return out
    }

    private fun drawSegTint(c: Canvas, clean: SegPostProcess.Clean) {
        // Cheap tint: stipple set pixels (full per-pixel draw would be slow).
        val w = clean.w; val h = clean.h
        val step = 2
        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                val i = y * w + x
                if (clean.pupilMask[i]) c.fillPx(x, y, pupilTintPaint)
                else if (clean.irisMask[i]) c.fillPx(x, y, irisTintPaint)
                x += step
            }
            y += step
        }
    }

    private fun Canvas.fillPx(x: Int, y: Int, p: Paint) =
        drawRect(x.toFloat(), y.toFloat(), (x + 2).toFloat(), (y + 2).toFloat(), p)

    private fun drawEllipse(c: Canvas, f: EllipseFitter.Fit, p: Paint) {
        c.save()
        c.rotate(Math.toDegrees(f.radian).toFloat(), f.cx.toFloat(), f.cy.toFloat())
        c.drawOval(
            RectF((f.cx - f.w).toFloat(), (f.cy - f.h).toFloat(),
                  (f.cx + f.w).toFloat(), (f.cy + f.h).toFloat()), p)
        c.restore()
    }

    private fun drawArrow(c: Canvas, x0: Float, y0: Float, x1: Float, y1: Float) {
        c.drawLine(x0, y0, x1, y1, gazePaint)
        val ang = Math.atan2((y1 - y0).toDouble(), (x1 - x0).toDouble())
        val len = hypot((x1 - x0).toDouble(), (y1 - y0).toDouble())
        if (len < 4) return
        val head = 12.0
        for (s in doubleArrayOf(2.6, -2.6)) {
            val hx = (x1 - head * Math.cos(ang + s)).toFloat()
            val hy = (y1 - head * Math.sin(ang + s)).toFloat()
            c.drawLine(x1, y1, hx, hy, gazePaint)
        }
    }
}
