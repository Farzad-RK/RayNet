package com.raynet.eyepatch

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.hypot

/**
 * Active, CLOSED-LOOP, hybrid screen-target gaze calibration canvas.
 *
 * Each frame the host supplies the combined gaze + face geometry + device
 * orientation. [ScreenGeometry] turns those into a COARSE screen point (metric
 * ray-plane intersection); [GazeScreenCalibrator] learns the RESIDUAL coarse->true
 * correction (kappa, scale, foreshortening) from the dots. The IMU re-anchors the
 * coarse point between the slow camera frames so the live cursor stays responsive
 * to device tilt.
 *
 * Closed loop: a dot is accepted only when the (corrected) gaze is BOTH steady and
 * in the dot's vicinity, held for a dwell. The first dots bootstrap on steadiness
 * alone (no residual yet); a per-dot timeout prevents hard-stalls. Tap to start.
 */
class CalibrationView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    private enum class State { IDLE, RUNNING, DONE }

    private val residual = GazeScreenCalibrator()
    val isCalibrated get() = residual.isCalibrated

    var onStatus: ((String) -> Unit)? = null

    private var imu: ImuTracker? = null
    private var screenGeom: ScreenGeometry? = null
    private var geomW = 0; private var geomH = 0

    private class Sample(val gaze: EyeGeometry.V3, val geom: EyePatchPipeline.FaceGeom, val rCap: Mat3?)
    private var last: Sample? = null

    // 3x3 grid of normalized target positions (inset from the edges).
    private val targets: List<DoubleArray> = run {
        val coords = doubleArrayOf(0.12, 0.5, 0.88)
        val out = ArrayList<DoubleArray>()
        for (ny in coords) for (nx in coords) out.add(doubleArrayOf(nx, ny))
        out
    }

    private var state = State.IDLE
    private var targetIdx = 0
    private var targetStartT = Double.NaN
    private var holdStartT = Double.NaN
    private val holdSamples = ArrayList<DoubleArray>()       // coarse (u,v) during the dwell
    private val recent = ArrayDeque<DoubleArray>()           // [t, u, v] within windowS
    private var cursor: DoubleArray? = null

    // Tunables. Sized for the ~4 fps camera cadence: short windows / high sample
    // counts are unreachable at 250 ms/frame, which previously wedged dot #1.
    private val stableRad = 0.05     // max gaze-angle spread (rad, ~3°) to count as a fixation
    private val vicinity = 0.16      // corrected-gaze distance to count as "on the dot"
    private val dwellS = 0.6         // hold the gate open this long to accept
    private val timeoutS = 8.0       // hard-accept after this even if never steady
    private val windowS = 1.0        // fixation-spread window (holds ~4 frames @4fps)
    private val minHold = 2          // min dwell samples (reachable in 0.6 s @4fps)

    // Auto-recalibration: the residual is fit at one viewing distance (iod_px); a
    // sustained distance change beyond the threshold invalidates it, so we re-run
    // the sequence automatically.
    private val distThreshold = 0.18    // relative iod_px change that triggers recalib
    private val distSustainS = 2.0      // must persist this long (ignores leaning)
    private val iodEmaAlpha = 0.25
    private var calibIod = 0.0           // reference iod_px at calibration time
    private var iodEma = 0.0
    private var driftStartT = Double.NaN
    private val iodRun = ArrayList<Double>()   // iod_px collected during a run

    // WHITE theme: the canvas is mostly white so it acts as a fill light for the
    // front camera; dots/cursor/text are dark for contrast.
    private val bgColor = Color.WHITE
    private val targetIdle = fill(Color.rgb(205, 205, 212))
    private val targetActive = fill(Color.rgb(220, 45, 45))
    private val targetDone = fill(Color.rgb(40, 150, 75))
    private val ringPaint = stroke(Color.rgb(120, 120, 128), 4f)
    private val holdRing = stroke(Color.rgb(230, 160, 0), 6f)
    private val cursorPaint = fill(Color.rgb(30, 110, 220))
    private val cursorRing = stroke(Color.rgb(30, 110, 220), 3f)
    private val textPaint = Paint().apply {
        color = Color.rgb(40, 40, 40); textSize = 38f; isAntiAlias = true; textAlign = Paint.Align.CENTER
    }
    private val hintPaint = Paint().apply {
        color = Color.rgb(90, 90, 90); textSize = 30f; isAntiAlias = true; textAlign = Paint.Align.CENTER
    }

    private fun fill(c: Int) = Paint().apply { color = c; style = Paint.Style.FILL; isAntiAlias = true }
    private fun stroke(c: Int, w: Float) =
        Paint().apply { color = c; style = Paint.Style.STROKE; strokeWidth = w; isAntiAlias = true }

    init {
        setOnClickListener {
            when (state) {
                State.IDLE, State.DONE -> start()
                State.RUNNING -> cancel()
            }
        }
    }

    /** Wire the high-rate IMU so the cursor re-anchors between camera frames. */
    fun attachImu(tracker: ImuTracker) {
        imu = tracker
        tracker.addListener { onImuTick() }
    }

    fun start() {
        residual.reset()
        holdSamples.clear(); recent.clear(); iodRun.clear()
        targetIdx = 0
        targetStartT = Double.NaN; holdStartT = Double.NaN; driftStartT = Double.NaN
        cursor = null
        state = State.RUNNING
        onStatus?.invoke("Look at the red dot and hold still…")
        invalidate()
    }

    fun cancel() {
        state = if (residual.isCalibrated) State.DONE else State.IDLE
        onStatus?.invoke(idleHint())
        invalidate()
    }

    /**
     * One camera frame. @param gaze combined camera-frame gaze (null = not tracking),
     * @param geom face geometry for the ray origin, @param rCap device orientation at
     * capture, @param t timestamp (s).
     */
    fun onSample(
        gaze: EyeGeometry.V3?, geom: EyePatchPipeline.FaceGeom?, rCap: Mat3?, t: Double,
    ) {
        ensureScreenGeom()
        if (gaze == null || geom == null) {
            last = null
            if (state == State.RUNNING) {
                onStatus?.invoke("No gaze — finish the look-around calibration on the Camera tab first.")
                holdStartT = Double.NaN; holdSamples.clear()
            }
            invalidate(); return
        }
        val s = Sample(gaze, geom, rCap); last = s
        val coarse = computeCoarse(s, applyDelta = false)
        cursor = if (residual.isCalibrated && coarse != null) residual.predict(coarse) else null
        when (state) {
            State.RUNNING -> { iodRun.add(geom.iodPx); runFsm(coarse, gaze, t) }
            State.DONE -> checkDistanceDrift(geom.iodPx, t)
            State.IDLE -> {}
        }
        invalidate()
    }

    /** Auto-recalibrate when the viewing distance has shifted past the threshold. */
    private fun checkDistanceDrift(iodPx: Double, t: Double) {
        if (calibIod <= 0.0 || iodPx <= 0.0) return
        iodEma = if (iodEma <= 0.0) iodPx else iodEma + iodEmaAlpha * (iodPx - iodEma)
        val rel = kotlin.math.abs(iodEma - calibIod) / calibIod
        if (rel > distThreshold) {
            if (driftStartT.isNaN()) driftStartT = t
            else if (t - driftStartT >= distSustainS) {
                onStatus?.invoke("Distance changed — recalibrating…")
                start()
            }
        } else driftStartT = Double.NaN
    }

    /**
     * Map a live gaze sample to the calibrated normalized screen point, for reuse
     * by the minigame. Uses the persisted screen geometry from calibration (the
     * game canvas shares the same content area), so it works even while this view
     * is hidden. Returns null until a calibration exists.
     */
    fun mapToScreen(gaze: EyeGeometry.V3?, geom: EyePatchPipeline.FaceGeom?, rCap: Mat3?): DoubleArray? {
        if (gaze == null || geom == null || !residual.isCalibrated || screenGeom == null) return null
        val coarse = computeCoarse(Sample(gaze, geom, rCap), applyDelta = false) ?: return null
        return residual.predict(coarse)
    }

    private fun onImuTick() {
        val s = last ?: return
        if (!residual.isCalibrated) return
        val coarse = computeCoarse(s, applyDelta = true) ?: return
        cursor = residual.predict(coarse)
        invalidate()
    }

    // --- gating state machine (camera cadence) ------------------------------
    private fun runFsm(coarse: DoubleArray?, gaze: EyeGeometry.V3, t: Double) {
        if (targetStartT.isNaN()) targetStartT = t
        val timedOut = (t - targetStartT) >= timeoutS
        if (coarse == null) {
            onStatus?.invoke("Look toward the screen…")
            holdStartT = Double.NaN; holdSamples.clear(); return
        }
        // Steadiness is judged on the RAW gaze angles (the ray-plane projection
        // amplifies that jitter, so coarse-point spread reads as never-steady).
        pushRecent(t, GazeScreenCalibrator.angles(gaze))
        val tgt = targets[targetIdx]
        val steady = recent.size >= 2 && spread() <= stableRad
        val corrected = if (residual.isCalibrated) residual.predict(coarse) else null
        // Vicinity only once a residual exists (corrected point); bootstrap on steadiness.
        val near = if (corrected != null) dist(corrected, tgt) <= vicinity else true
        // The timeout is a HARD accept — it opens the gate regardless of steadiness
        // so a noisy model / shaky fixation can never wedge the sequence.
        val gateOpen = (steady && near) || timedOut

        if (gateOpen) {
            if (holdStartT.isNaN()) { holdStartT = t; holdSamples.clear() }
            holdSamples.add(coarse)
            onStatus?.invoke(if (timedOut && !(steady && near)) "Holding (model unsure)…" else "Hold steady…")
            val held = t - holdStartT
            if ((held >= dwellS && holdSamples.size >= minHold) ||
                (timedOut && holdSamples.isNotEmpty())) acceptTarget()
        } else {
            holdStartT = Double.NaN; holdSamples.clear()
            onStatus?.invoke(
                if (!steady) "Look at the red dot and hold still…"
                else "Move your gaze onto the red dot…")
        }
    }

    private fun acceptTarget() {
        val tgt = targets[targetIdx]
        residual.addTarget(tgt[0], tgt[1], holdSamples.toList())
        holdSamples.clear(); holdStartT = Double.NaN; recent.clear()
        if (residual.targetCount >= 3) residual.fit()          // provisional refit
        targetIdx++
        targetStartT = Double.NaN
        if (targetIdx >= targets.size) {
            val ok = residual.fit()
            // Reference viewing distance for the auto-recalibration trigger.
            calibIod = if (iodRun.isNotEmpty()) iodRun.sorted()[iodRun.size / 2] else 0.0
            iodEma = calibIod; driftStartT = Double.NaN
            state = State.DONE
            onStatus?.invoke(if (ok) "Calibrated — the blue dot follows your gaze. Tap to redo."
                             else "Calibration failed — tap to retry.")
        }
    }

    // --- geometry helpers ---------------------------------------------------
    private fun computeCoarse(s: Sample, applyDelta: Boolean): DoubleArray? {
        val sg = screenGeom ?: return null
        val dR = if (applyDelta && s.rCap != null)
            imu?.current()?.let { it.transpose() * s.rCap } else null
        return sg.project(
            s.geom.irisMidX, s.geom.irisMidY, s.geom.iodPx,
            s.geom.frameW, s.geom.frameH, s.gaze, dR)
    }

    private fun ensureScreenGeom() {
        if (width == 0 || height == 0) return
        if (screenGeom != null && geomW == width && geomH == height) return
        val dm = resources.displayMetrics
        val loc = IntArray(2); getLocationOnScreen(loc)
        screenGeom = ScreenGeometry(
            viewWidthPx = width, viewHeightPx = height,
            viewLeftPx = loc[0], viewTopPx = loc[1],
            pxPerMmX = dm.xdpi / 25.4, pxPerMmY = dm.ydpi / 25.4,
            camXpx = dm.widthPixels / 2.0, camYpx = 0.0,   // front camera ~ top-center
        )
        geomW = width; geomH = height
    }

    private fun pushRecent(t: Double, c: DoubleArray) {
        recent.addLast(doubleArrayOf(t, c[0], c[1]))
        while (recent.isNotEmpty() && t - recent.first()[0] > windowS) recent.removeFirst()
    }

    private fun spread(): Double {
        var minU = Double.MAX_VALUE; var maxU = -Double.MAX_VALUE
        var minV = Double.MAX_VALUE; var maxV = -Double.MAX_VALUE
        for (r in recent) {
            if (r[1] < minU) minU = r[1]; if (r[1] > maxU) maxU = r[1]
            if (r[2] < minV) minV = r[2]; if (r[2] > maxV) maxV = r[2]
        }
        return maxOf(maxU - minU, maxV - minV)
    }

    private fun dist(p: DoubleArray, q: DoubleArray) = hypot(p[0] - q[0], p[1] - q[1])

    private fun idleHint() =
        if (residual.isCalibrated) "Tap to recalibrate the screen mapping."
        else "Tap to start screen calibration. Keep your head still and follow the dots."

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(bgColor)
        val w = width.toFloat(); val h = height.toFloat()
        val r = (minOf(w, h) * 0.022f).coerceAtLeast(10f)

        for (i in targets.indices) {
            val cx = (targets[i][0] * w).toFloat()
            val cy = (targets[i][1] * h).toFloat()
            val active = state == State.RUNNING && i == targetIdx
            when {
                state == State.DONE || i < targetIdx -> canvas.drawCircle(cx, cy, r * 0.6f, targetDone)
                active -> {
                    canvas.drawCircle(cx, cy, r, targetActive)
                    canvas.drawCircle(cx, cy, r * 1.7f, ringPaint)
                    if (!holdStartT.isNaN()) canvas.drawCircle(cx, cy, r * 2.3f, holdRing)
                    canvas.drawCircle(cx, cy, (vicinity * minOf(w, h)).toFloat(), ringPaint)
                }
                else -> canvas.drawCircle(cx, cy, r * 0.6f, targetIdle)
            }
        }

        cursor?.let {
            val cx = (it[0] * w).toFloat(); val cy = (it[1] * h).toFloat()
            canvas.drawCircle(cx, cy, r * 0.8f, cursorPaint)
            canvas.drawCircle(cx, cy, r * 1.6f, cursorRing)
        }

        if (state == State.IDLE) {
            canvas.drawText(if (residual.isCalibrated) "Screen calibration ready"
                            else "Screen calibration", w / 2, h / 2 - 24f, textPaint)
            canvas.drawText("Tap anywhere to begin", w / 2, h / 2 + 28f, hintPaint)
        } else if (state == State.RUNNING) {
            canvas.drawText("${targetIdx + 1} / ${targets.size}", w / 2, h * 0.93f, hintPaint)
        }
    }
}
