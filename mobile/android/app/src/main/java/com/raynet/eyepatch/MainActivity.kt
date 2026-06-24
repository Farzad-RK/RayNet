package com.raynet.eyepatch

import android.Manifest
import android.app.AlertDialog
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.os.SystemClock
import android.view.View
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import android.util.Size
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.SurfaceOrientedMeteringPointFactory
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import java.util.concurrent.TimeUnit
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

/**
 * Camera -> EyePatchPipeline -> annotated composite, with a calibrate/track UI.
 *
 * Model loading (52 MB head-pose .ort + seg .ort + MediaPipe) happens on the
 * analysis thread, NOT in onCreate — a missing/!broken asset surfaces as a
 * banner message instead of crashing the Activity. All pipeline access is
 * serialized on that single executor.
 */
class MainActivity : ComponentActivity() {

    private lateinit var imageView: ImageView
    private lateinit var calibrationView: CalibrationView
    private lateinit var banner: TextView
    private lateinit var progress: ProgressBar
    private lateinit var btnCalibrate: Button
    private lateinit var btnReset: Button
    private lateinit var btnTint: Button
    private lateinit var btnCamera: Button
    private lateinit var tabCamera: TextView
    private lateinit var tabCalib: TextView
    private lateinit var tabGame: TextView
    private lateinit var cameraControls: View
    private lateinit var gameView: GameView
    private lateinit var imu: ImuTracker

    private enum class Tab { CAMERA, CALIB, GAME }
    @Volatile private var activeTab = Tab.CAMERA

    private val analysisExecutor = Executors.newSingleThreadExecutor()

    @Volatile private var pipeline: EyePatchPipeline? = null
    @Volatile private var initError: String? = null
    @Volatile private var calibrating = false
    private val calibTarget = 30   // valid frames per eye before auto-fitting
                                   // (eyes alternate, so ~2x frames total)

    private val requestCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else banner.text = "Camera permission denied — cannot run."
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // The bright/white UI doubles as a fill light for the front camera — keep the
        // screen on and force max brightness so it actually illuminates the face.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.attributes = window.attributes.apply { screenBrightness = 1.0f }
        setContentView(R.layout.activity_main)
        imageView = findViewById(R.id.imageView)
        calibrationView = findViewById(R.id.calibrationView)
        banner = findViewById(R.id.statusBanner)
        progress = findViewById(R.id.calibProgress)
        btnCalibrate = findViewById(R.id.btnCalibrate)
        btnReset = findViewById(R.id.btnReset)
        btnTint = findViewById(R.id.btnTint)
        btnCamera = findViewById(R.id.btnCamera)
        tabCamera = findViewById(R.id.tabCamera)
        tabCalib = findViewById(R.id.tabCalib)
        tabGame = findViewById(R.id.tabGame)
        cameraControls = findViewById(R.id.cameraControls)
        gameView = findViewById(R.id.gameView)

        imu = ImuTracker(this)
        calibrationView.attachImu(imu)

        tabCamera.setOnClickListener { selectTab(Tab.CAMERA) }
        tabCalib.setOnClickListener { selectTab(Tab.CALIB) }
        tabGame.setOnClickListener { selectTab(Tab.GAME) }
        calibrationView.onStatus = { msg -> runOnUiThread { if (activeTab == Tab.CALIB) banner.text = msg } }
        gameView.onStatus = { msg -> runOnUiThread { if (activeTab == Tab.GAME) banner.text = msg } }
        selectTab(Tab.CAMERA)

        btnCamera.setOnClickListener { showCameraSettings() }
        btnCalibrate.setOnClickListener { onCalibrate() }
        btnReset.setOnClickListener {
            calibrating = false
            analysisExecutor.execute { pipeline?.resetTracking() }
            runOnUiThread { progress.visibility = View.GONE; banner.text = readyText() }
        }
        btnTint.setOnClickListener {
            val p = pipeline ?: return@setOnClickListener
            p.showSegTint = !p.showSegTint
            btnTint.text = if (p.showSegTint) "Masks: On" else "Masks: Off"
        }
        setControlsEnabled(false)

        // Build the pipeline off the main thread.
        analysisExecutor.execute {
            try {
                pipeline = EyePatchPipeline(this)
                runOnUiThread { banner.text = readyText(); setControlsEnabled(true) }
            } catch (e: Throwable) {
                initError = e.message ?: e.toString()
                runOnUiThread { banner.text = "Model load failed: $initError" }
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) startCamera()
        else requestCamera.launch(Manifest.permission.CAMERA)
    }

    private fun onCalibrate() {
        if (pipeline == null) return
        calibrating = true
        analysisExecutor.execute { pipeline?.startCalibration() }
        runOnUiThread {
            progress.visibility = View.VISIBLE; progress.progress = 0
            banner.text = "Calibrating — slowly roll your eyes around in a circle…"
        }
    }

    private fun setControlsEnabled(on: Boolean) {
        btnCalibrate.isEnabled = on; btnReset.isEnabled = on; btnTint.isEnabled = on
    }

    // The camera frame keeps analyzing on every tab (the calibration canvas and the
    // game both need the live gaze); we only swap which view + controls are visible.
    private fun selectTab(tab: Tab) {
        activeTab = tab
        val cam = tab == Tab.CAMERA
        imageView.visibility = if (cam) View.VISIBLE else View.GONE
        calibrationView.visibility = if (tab == Tab.CALIB) View.VISIBLE else View.GONE
        gameView.visibility = if (tab == Tab.GAME) View.VISIBLE else View.GONE
        cameraControls.visibility = if (cam) View.VISIBLE else View.GONE
        btnCamera.visibility = if (cam) View.VISIBLE else View.GONE
        if (cam) { imu.stop(); progress.visibility = progress.visibility } else imu.start()
        pipeline?.renderComposite = cam          // skip the unused composite off the camera tab

        val sel = 0xFFD7E3FF.toInt(); val unsel = 0xFFFFFFFF.toInt()   // light theme
        tabCamera.setBackgroundColor(if (cam) sel else unsel)
        tabCalib.setBackgroundColor(if (tab == Tab.CALIB) sel else unsel)
        tabGame.setBackgroundColor(if (tab == Tab.GAME) sel else unsel)
        banner.text = when (tab) {
            Tab.CAMERA -> readyText()
            Tab.CALIB -> if (calibrationView.isCalibrated) "Tap to recalibrate the screen mapping."
                         else "Tap to start screen calibration. Keep your head still and follow the dots."
            Tab.GAME -> if (calibrationView.isCalibrated) "Look at a shape to pick it up, then the ring to place it."
                        else "Calibrate on the Calibration tab first."
        }
    }

    private fun readyText() =
        "Ready. Tap Calibrate, then slowly look around in circles to fit your eye model."

    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    // User-selectable camera config (defaults = previous hardcoded values).
    // lensFacing uses CameraCharacteristics.LENS_FACING_* constants.
    private var lensFacing = CameraCharacteristics.LENS_FACING_FRONT
    private var targetSize = Size(1280, 960)

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            cameraProvider = future.get()
            bindCamera()
        }, ContextCompat.getMainExecutor(this))
    }

    /** (Re)bind the analysis use case using the current lens + resolution. */
    private fun bindCamera() {
        val provider = cameraProvider ?: return
        // HIGH-RES analysis stream: eye patches are sliced from these full
        // pixels (not the old ~480px frame), so iris/pupil come out sharp.
        val resSel = ResolutionSelector.Builder()
            // CRITICAL: ImageAnalysis defaults to PREFER_CAPTURE_RATE, which caps the
            // analysis stream at 640x480 and silently ignores the requested size — the
            // eye then comes through at ~60px and the warp upsamples it ~4x (blurry
            // patches). Preferring resolution unlocks the selected 1280x960 so the eye
            // arrives at ~170px and the patch is a near-1:1 crop.
            .setAllowedResolutionMode(
                ResolutionSelector.PREFER_HIGHER_RESOLUTION_OVER_CAPTURE_RATE)
            .setResolutionStrategy(
                ResolutionStrategy(targetSize,
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
            .build()
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(resSel)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        analysis.setAnalyzer(analysisExecutor) { proxy -> analyze(proxy) }
        val selector = CameraSelector.Builder()
            .requireLensFacing(
                if (lensFacing == CameraCharacteristics.LENS_FACING_BACK)
                    CameraSelector.LENS_FACING_BACK
                else CameraSelector.LENS_FACING_FRONT)
            .build()
        provider.unbindAll()
        camera = try {
            provider.bindToLifecycle(this, selector, analysis).also {
                android.util.Log.i("RayNetCam",
                    "requested=$targetSize bound=${analysis.resolutionInfo?.resolution}")
            }
        } catch (e: Exception) {
            runOnUiThread { banner.text = "Camera bind failed: ${e.message}" }
            null
        }
    }

    private fun facingLabel(facing: Int) = when (facing) {
        CameraCharacteristics.LENS_FACING_FRONT -> "Front"
        CameraCharacteristics.LENS_FACING_BACK -> "Back"
        else -> "External"
    }

    /**
     * Dialog to pick the camera lens and the analysis resolution. Both lists are
     * enumerated from the device itself (Camera2 characteristics), so only valid
     * choices are offered; "Apply" rebinds the camera live.
     */
    private fun showCameraSettings() {
        val provider = cameraProvider ?: run {
            banner.text = "Camera not ready yet."
            return
        }
        // facing -> supported YUV analysis sizes (first camera of each facing).
        val facingToSizes = LinkedHashMap<Int, List<Size>>()
        for (info in provider.availableCameraInfos) {
            val c2 = try { Camera2CameraInfo.from(info) } catch (_: Throwable) { continue }
            val facing = c2.getCameraCharacteristic(CameraCharacteristics.LENS_FACING) ?: continue
            if (facingToSizes.containsKey(facing)) continue
            val map = c2.getCameraCharacteristic(
                CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            val sizes = map?.getOutputSizes(ImageFormat.YUV_420_888)?.toList()
                ?.sortedByDescending { it.width.toLong() * it.height } ?: emptyList()
            facingToSizes[facing] = sizes
        }
        if (facingToSizes.isEmpty()) { banner.text = "No cameras available."; return }
        val facings = facingToSizes.keys.toList()

        val lensSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                facings.map { facingLabel(it) })
        }
        val resSpinner = Spinner(this)

        fun loadSizes(facing: Int) {
            val sizes = facingToSizes[facing].orEmpty()
            resSpinner.adapter = ArrayAdapter(this,
                android.R.layout.simple_spinner_dropdown_item,
                sizes.map { "${it.width} × ${it.height}" })
            val idx = sizes.indexOfFirst {
                it.width == targetSize.width && it.height == targetSize.height
            }
            if (idx >= 0) resSpinner.setSelection(idx)
        }

        val curIdx = facings.indexOf(lensFacing).coerceAtLeast(0)
        lensSpinner.setSelection(curIdx)
        loadSizes(facings[curIdx])
        lensSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) =
                loadSizes(facings[pos])
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }

        val pad = (16 * resources.displayMetrics.density).toInt()
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, 0)
            addView(TextView(this@MainActivity).apply { text = "Camera" })
            addView(lensSpinner)
            addView(TextView(this@MainActivity).apply { text = "Analysis resolution" })
            addView(resSpinner)
        }

        AlertDialog.Builder(this)
            .setTitle("Camera settings")
            .setView(container)
            .setPositiveButton("Apply") { _, _ ->
                val facing = facings[lensSpinner.selectedItemPosition]
                val sizes = facingToSizes[facing].orEmpty()
                val size = sizes.getOrNull(resSpinner.selectedItemPosition) ?: targetSize
                if (facing != lensFacing) {
                    // New lens may not support the old eye-model scale — start fresh.
                    calibrating = false
                    analysisExecutor.execute { pipeline?.resetTracking() }
                }
                lensFacing = facing
                targetSize = size
                bindCamera()
                runOnUiThread { progress.visibility = View.GONE; banner.text = readyText() }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun analyze(proxy: ImageProxy) {
        try {
            val p = pipeline ?: return
            val upright = proxy.toUprightBitmap()
            val t = SystemClock.elapsedRealtimeNanos() * 1e-9
            val result = p.process(upright, t)

            // Calibration auto-finish on this (pipeline) thread.
            if (calibrating && p.minCalibCount() >= calibTarget) {
                val ok = p.finishCalibration()
                calibrating = false
                runOnUiThread {
                    if (activeTab == Tab.CAMERA) {
                        progress.visibility = View.GONE
                        banner.text = if (ok) "Tracking — gaze is live. (scale uncalibrated)"
                                      else "Calibration failed (no stable pupil). Try again."
                    }
                }
            }
            if (result.faceFound) maybeFocus()

            when (activeTab) {
                Tab.CALIB -> {
                    // Feed the screen-calibration canvas the live binocular gaze + the
                    // face geometry and device orientation for the metric intersection.
                    val gaze = p.combinedGazeCam()
                    val geom = result.faceGeom
                    val rDev = imu.current()
                    runOnUiThread { calibrationView.onSample(gaze, geom, rDev, t) }
                }
                Tab.GAME -> {
                    // Reuse the calibrated mapping to drive the game cursor.
                    val cursor = calibrationView.mapToScreen(p.combinedGazeCam(), result.faceGeom, imu.current())
                    runOnUiThread { gameView.onCursor(cursor, t) }
                }
                Tab.CAMERA -> {
                    renderUi(p, result)
                    runOnUiThread { imageView.setImageBitmap(result.composite) }
                }
            }
        } finally {
            proxy.close()
        }
    }

    // Periodically drive AF+AE metering onto the central ROI (where the centered
    // face/eyes sit). Center (0.5,0.5) is rotation/mirror-invariant, so it targets
    // the eyes reliably without per-device coordinate mapping. Many front cameras
    // are fixed-focus (AF is then a no-op) but AE/AWB metering still sharpens the
    // eye exposure. Best-effort — failures are ignored.
    private var lastFocusMs = 0L
    private fun maybeFocus() {
        val cam = camera ?: return
        val now = SystemClock.elapsedRealtime()
        if (now - lastFocusMs < 2500) return
        lastFocusMs = now
        try {
            val pt = SurfaceOrientedMeteringPointFactory(1f, 1f).createPoint(0.5f, 0.45f)
            val action = FocusMeteringAction.Builder(
                pt, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE)
                .setAutoCancelDuration(4, TimeUnit.SECONDS)
                .build()
            cam.cameraControl.startFocusAndMetering(action)
        } catch (_: Throwable) { /* unsupported / fixed-focus */ }
    }

    private fun renderUi(p: EyePatchPipeline, r: EyePatchPipeline.Result) {
        if (calibrating) {
            val prog = (100 * p.minCalibCount() / calibTarget).coerceIn(0, 100)
            runOnUiThread { progress.progress = prog }
            return
        }
        if (p.anyTracking()) return  // banner already says "Tracking"
        val msg = when {
            !r.faceFound -> "No face detected — center your face in the frame."
            else -> readyText()
        }
        runOnUiThread { if (banner.text != msg) banner.text = msg }
    }

    /** ImageProxy (CameraX 1.3+) -> Bitmap rotated to upright. */
    private fun ImageProxy.toUprightBitmap(): Bitmap {
        val bmp = toBitmap()
        val deg = imageInfo.rotationDegrees
        if (deg == 0) return bmp
        val m = Matrix().apply { postRotate(deg.toFloat()) }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }

    override fun onDestroy() {
        super.onDestroy()
        imu.stop()
        val p = pipeline; pipeline = null
        analysisExecutor.execute { p?.close() }
        analysisExecutor.shutdown()
    }
}
