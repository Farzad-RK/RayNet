package com.raynet.eyepatch

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log

/**
 * Builds an [OrtSession] with a hardware-accelerated execution provider and a
 * safe fallback to plain CPU. The head-pose and segmentation conv nets dominate
 * the frame budget (~940 ms of ~1046 ms on the bare CPU EP), so the EP choice is
 * the single biggest real-time lever.
 *
 *  - XNNPACK: CPU-side, ARM-optimized conv kernels; reliably available in the
 *    `onnxruntime-android` package and the safest big win.
 *  - NNAPI: offloads to GPU/DSP/NPU; can be much faster on capable devices but
 *    falls back per-op and is device-dependent — try it, measure, keep if faster.
 *
 * On any EP failure the session is rebuilt on CPU so the app never crashes for an
 * unsupported accelerator.
 */
object OrtAccel {
    enum class Mode { CPU, XNNPACK, NNAPI }

    /** Switchable at runtime (e.g. from a debug toggle) before sessions are built. */
    // Measured on a Snapdragon/MIUI device: XNNPACK(4t) was SLOWER and NNAPI fell
    // back to the slow "nnapi-reference" CPU driver (no GPU/NPU offload). Plain CPU
    // EP was fastest, so it is the default; switch to try others per device.
    @Volatile var mode: Mode = Mode.CPU

    fun createSession(env: OrtEnvironment, modelBytes: ByteArray, threads: Int = 4): OrtSession {
        val attempts = if (mode == Mode.CPU) listOf(Mode.CPU) else listOf(mode, Mode.CPU)
        var last: Throwable? = null
        for (m in attempts) {
            try {
                val o = OrtSession.SessionOptions().apply {
                    addConfigEntry("session.load_model_format", "ORT")
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    setIntraOpNumThreads(threads)
                    when (m) {
                        Mode.XNNPACK -> addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                        Mode.NNAPI -> addNnapi()
                        Mode.CPU -> {}
                    }
                }
                val s = env.createSession(modelBytes, o)
                Log.i("OrtAccel", "session created EP=$m threads=$threads")
                return s
            } catch (e: Throwable) {
                last = e
                Log.w("OrtAccel", "EP=$m failed (${e.message}); falling back")
            }
        }
        throw last ?: IllegalStateException("no EP succeeded")
    }
}
