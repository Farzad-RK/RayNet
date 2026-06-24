package com.raynet.eyepatch

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import java.util.concurrent.Callable
import java.util.concurrent.Executors

/**
 * End-to-end on-device pipeline, mirroring sixdrepnet/demo.py + the 3DeepVOG
 * gaze stage:
 *
 *   FaceLandmarker -> iris crop -> ONNX head pose -> One Euro smoothing
 *   -> head-pose-normalized eye patches -> per-eye [EyeTracker] (segment, ellipse
 *      fit, calibrate/predict) -> [GazeOverlay] -> composite canvas.
 *
 * `process()` returns a [Result]: the composite Bitmap (face + two annotated eye
 * patches) plus the per-eye tracker results, so the Activity can drive the
 * calibration UI. The gaze stage runs only once both eyes have patches.
 */
class EyePatchPipeline(context: Context) {

    private val landmarker = FaceLandmarkerHelper(context)
    private val cropper = FaceCropper()
    private val pose = HeadPoseEstimator(context)
    private val smoother = HeadPoseSmoother(minCutoff = 1.0, beta = 0.3)
    private val eyeNorm = EyeNormalizer()

    private val segmenter = EyeSegmenter(context)
    private val trackers = mapOf(
        "right" to EyeTracker(segmenter, "right"),
        "left" to EyeTracker(segmenter, "left"),
    )

    private var missCount = 0
    private val maxMissBeforeReset = 5

    private val patchW = 320
    private val patchH = 240

    // --- real-time throttling (the ONNX nets are CPU-bound; see logTiming) ---
    // Head pose (ONNX, ~300 ms/run) used to run INLINE every Nth frame, stalling
    // the gaze pipeline periodically. It now runs ASYNC on its own thread: the
    // analysis loop reads the latest smoothed rotation and submits a fresh estimate
    // whenever the pose thread is free (head moves slowly, so a slightly stale R is
    // fine — the warp already reused it across frames). Segmentation is the dominant
    // cost, so we segment ONE eye per frame (alternating).
    private val poseExecutor = Executors.newSingleThreadExecutor()
    @Volatile private var poseBusy = false
    // Landmarks run on a downscaled copy this wide; patches are sliced from the
    // full-res frame (see process()).
    private val detectWidth = 480
    private var frameIdx = 0
    @Volatile private var cachedR: Mat3? = null
    private val lastResult = HashMap<String, EyeTracker.Result>()

    /** Skip the (unused) composite/overlay rendering when a non-camera tab is up. */
    @Volatile var renderComposite = true
    private val placeholder1x1 = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)

    /** When true, the seg masks are tinted on the patches (toggleable from UI). */
    @Volatile var showSegTint = false

    class Result(
        val composite: Bitmap,
        val faceFound: Boolean,
        val left: EyeTracker.Result?,
        val right: EyeTracker.Result?,
        val faceGeom: FaceGeom? = null,
    )

    /** 2D face geometry the screen-gaze stage needs to build the gaze ray. */
    class FaceGeom(
        val irisMidX: Double, val irisMidY: Double, val iodPx: Double,
        val frameW: Int, val frameH: Int,
    )

    // --- calibration API (delegates to both eyes) ---------------------------
    fun startCalibration() = trackers.values.forEach { it.startCalibration() }
    fun finishCalibration(): Boolean {
        // succeeds if at least one eye solves a model
        var any = false
        trackers.values.forEach { if (it.finishCalibration()) any = true }
        return any
    }
    fun resetTracking() = trackers.values.forEach { it.reset() }
    fun minCalibCount(): Int = trackers.values.minOf { it.calibCount }
    fun anyCalibrating(): Boolean = trackers.values.any { it.mode == EyeTracker.Mode.CALIBRATING }
    fun anyTracking(): Boolean = trackers.values.any { it.mode == EyeTracker.Mode.TRACKING }

    /**
     * Combined binocular gaze in the common camera frame (mean of whichever eyes
     * are currently tracking), or null if neither eye has a gaze yet. This is the
     * single direction the screen-gaze calibration consumes.
     */
    fun combinedGazeCam(): EyeGeometry.V3? {
        var sx = 0.0; var sy = 0.0; var sz = 0.0; var n = 0
        for (t in trackers.values) {
            val g = t.currentGazeCam ?: continue
            sx += g.x; sy += g.y; sz += g.z; n++
        }
        if (n == 0) return null
        val v = EyeGeometry.V3(sx / n, sy / n, sz / n)
        return if (v.norm() < 1e-9) null else v.normalized()
    }

    /**
     * @param frame the HIGH-RESOLUTION upright camera frame. Landmarks are
     *   detected on a downscaled copy (fast), but ALL pixel sampling — the head-
     *   pose face crop and the eye-patch warp — reads `frame` at full resolution,
     *   so the 320x240 patches are sliced from sharp pixels, not upsampled from a
     *   tiny low-res eye region. The on-screen composite uses the small copy.
     */
    fun process(frame: Bitmap, t: Double): Result {
        val tStart = android.os.SystemClock.elapsedRealtimeNanos()
        val lo = downscaleForDetect(frame)            // small frame for MediaPipe + display
        val hiToLo = lo.width.toDouble() / frame.width
        val lms = landmarker.detect(lo, frame.width, frame.height)   // coords in HI space
        val tLm = android.os.SystemClock.elapsedRealtimeNanos()
        if (lms == null) {
            if (++missCount >= maxMissBeforeReset) { smoother.reset(); eyeNorm.reset(); cachedR = null }
            return Result(composite(lo, emptyMap(), null), false, null, null)
        }
        missCount = 0
        frameIdx++

        val crop = cropper.compute(lms, frame.width, frame.height)
            ?: return Result(composite(lo, emptyMap(), null), true, null, null)
        val box = crop.box                            // HI coords
        if (box.width() <= 0 || box.height() <= 0)
            return Result(composite(lo, emptyMap(), null), true, null, null)
        val boxLo = scaleRect(box, hiToLo)            // for drawing on the small frame

        // Head pose: async. Bootstrap blocks ONCE (we need an initial R to warp);
        // after that we submit a fresh estimate whenever the pose thread is idle and
        // never block the gaze loop on it.
        if (cachedR == null) {
            val faceCrop = Bitmap.createBitmap(frame, box.left, box.top, box.width(), box.height())
            cachedR = poseExecutor.submit(Callable { smoother.filter(pose.estimate(faceCrop), t) }).get()
        } else if (!poseBusy) {
            poseBusy = true
            val faceCrop = Bitmap.createBitmap(frame, box.left, box.top, box.width(), box.height())
            poseExecutor.execute {
                try { cachedR = smoother.filter(pose.estimate(faceCrop), t) }
                finally { poseBusy = false }
            }
        }
        val rSmoothed = cachedR!!
        val tPose = android.os.SystemClock.elapsedRealtimeNanos()
        val patches = eyeNorm.process(frame, lms, rSmoothed, t)   // warp from FULL-RES frame
        val tWarp = android.os.SystemClock.elapsedRealtimeNanos()

        // Tier 2/3: segment ONE eye this frame (alternating); the other reuses its
        // last result for the overlay. Skip seg entirely while IDLE (fast preview).
        val results = HashMap<String, EyeTracker.Result>()
        val annotated = HashMap<String, Bitmap>()
        val activeSide = if (frameIdx % 2 == 0) "left" else "right"
        for ((side, patch) in patches) {
            val tracker = trackers[side]!!
            // Binocular hint: the OTHER eye's gaze in the common camera frame.
            val otherCam = trackers[if (side == "left") "right" else "left"]?.currentGazeCam
            val res = if (side == activeSide && tracker.mode != EyeTracker.Mode.IDLE) {
                tracker.process(patch.bitmap, patch.rNorm, patch.flippedX, otherCam, rSmoothed)
                    .also { lastResult[side] = it }
            } else lastResult[side]
            if (res != null) results[side] = res
            if (renderComposite) {
                annotated[side] = if (res != null) GazeOverlay.render(patch.bitmap, res, showSegTint)
                                  else patch.bitmap
            }
        }
        val tEyes = android.os.SystemClock.elapsedRealtimeNanos()
        val faceGeom = FaceGeom(crop.irisMidX, crop.irisMidY, crop.iodPx, frame.width, frame.height)
        val compositeBmp = if (renderComposite) composite(lo, annotated, boxLo) else placeholder1x1
        val out = Result(compositeBmp, true, results["left"], results["right"], faceGeom)
        val tEnd = android.os.SystemClock.elapsedRealtimeNanos()
        logTiming(frame, tStart, tLm, tPose, tWarp, tEyes, tEnd, patches.size)
        return out
    }

    // --- timing instrumentation (averaged + logged every 30 frames) ---------
    private var nFrames = 0
    private var aLm = 0.0; private var aPose = 0.0; private var aWarp = 0.0
    private var aEyes = 0.0; private var aComp = 0.0; private var aTot = 0.0
    private fun logTiming(
        frame: Bitmap, t0: Long, tLm: Long, tPose: Long, tWarp: Long, tEyes: Long, tEnd: Long, nEyes: Int,
    ) {
        val ms = 1e-6
        aLm += (tLm - t0) * ms; aPose += (tPose - tLm) * ms; aWarp += (tWarp - tPose) * ms
        aEyes += (tEyes - tWarp) * ms; aComp += (tEnd - tEyes) * ms; aTot += (tEnd - t0) * ms
        if (++nFrames >= 30) {
            val n = nFrames.toDouble()
            android.util.Log.i("RayNetPerf", "in=%dx%d eyes=%d | landmark=%.0f pose=%.0f warp=%.0f tier2-3=%.0f composite=%.0f TOTAL=%.0fms (~%.1f fps)"
                .format(frame.width, frame.height, nEyes, aLm/n, aPose/n, aWarp/n, aEyes/n, aComp/n, aTot/n, 1000.0/(aTot/n)))
            nFrames = 0; aLm = 0.0; aPose = 0.0; aWarp = 0.0; aEyes = 0.0; aComp = 0.0; aTot = 0.0
        }
    }

    private fun downscaleForDetect(hi: Bitmap): Bitmap {
        if (hi.width <= detectWidth) return hi
        val h = (hi.height.toLong() * detectWidth / hi.width).toInt()
        return Bitmap.createScaledBitmap(hi, detectWidth, h, true)
    }

    private fun scaleRect(r: Rect, s: Double) = Rect(
        (r.left * s).toInt(), (r.top * s).toInt(), (r.right * s).toInt(), (r.bottom * s).toInt())

    fun close() {
        // Drain the pose thread before closing its ONNX session (closing mid-run crashes).
        poseExecutor.shutdown()
        try { poseExecutor.awaitTermination(2, java.util.concurrent.TimeUnit.SECONDS) } catch (_: InterruptedException) {}
        landmarker.close()
        pose.close()
        segmenter.close()
    }

    // --- Composite canvas: face on top, "right | left" patches below ---------
    private val labelPaint = Paint().apply {
        color = Color.DKGRAY; textSize = 28f; isAntiAlias = true
    }
    private val boxPaint = Paint().apply {
        color = Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 3f
    }
    private val placeholderPaint = Paint().apply {
        color = Color.DKGRAY; style = Paint.Style.STROKE; strokeWidth = 2f
    }

    private fun composite(frame: Bitmap, patches: Map<String, Bitmap>, box: Rect?): Bitmap {
        val pad = 16
        val labelH = 34
        val rowW = 2 * patchW + 3 * pad
        val cw = maxOf(frame.width, rowW)
        val ch = frame.height + pad + labelH + patchH + pad
        val canvas = Bitmap.createBitmap(cw, ch, Bitmap.Config.ARGB_8888)
        val c = Canvas(canvas)
        c.drawColor(Color.WHITE)   // white surround doubles as fill light

        val fx = (cw - frame.width) / 2
        c.drawBitmap(frame, fx.toFloat(), 0f, null)
        if (box != null) {
            c.drawRect(
                (fx + box.left).toFloat(), box.top.toFloat(),
                (fx + box.right).toFloat(), box.bottom.toFloat(), boxPaint,
            )
        }

        var x = (cw - rowW) / 2 + pad
        val y = frame.height + pad + labelH
        for (name in arrayOf("right", "left")) {
            c.drawText(name, x.toFloat(), (y - 8).toFloat(), labelPaint)
            val patch = patches[name]
            if (patch != null) {
                c.drawBitmap(patch, x.toFloat(), y.toFloat(), null)
            } else {
                c.drawRect(x.toFloat(), y.toFloat(),
                    (x + patchW).toFloat(), (y + patchH).toFloat(), placeholderPaint)
                c.drawText("no eye", (x + patchW / 2 - 40).toFloat(),
                    (y + patchH / 2).toFloat(), labelPaint)
            }
            x += patchW + pad
        }
        return canvas
    }
}
