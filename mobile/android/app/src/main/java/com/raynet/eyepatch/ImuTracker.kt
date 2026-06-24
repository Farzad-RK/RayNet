package com.raynet.eyepatch

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager

/**
 * Device orientation from the fused rotation vector (accelerometer + gyroscope +
 * magnetometer). Gives a gravity-and-heading-referenced device->world rotation at
 * high rate (~100-200 Hz), far faster than the ~4 fps camera pipeline.
 *
 * We use it for ROTATION only — translation is deliberately not derived here, since
 * double-integrating the accelerometer drifts to meters within seconds. The world
 * frame is Android's: X east, Y magnetic north, Z up.
 *
 * Listeners are notified on the main looper so they can touch Views directly. The
 * latest rotation is also readable synchronously from the camera thread.
 */
class ImuTracker(context: Context) : SensorEventListener {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val rotationSensor: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    val isAvailable get() = rotationSensor != null

    @Volatile private var rotation: Mat3? = null
    @Volatile var timestampNs: Long = 0L; private set

    private val r9 = FloatArray(9)
    private val listeners = ArrayList<() -> Unit>()

    /** device->world rotation, or null until the first sensor event. */
    fun current(): Mat3? = rotation

    /** Subscribe to high-rate updates (called on the main thread). */
    fun addListener(l: () -> Unit) { listeners.add(l) }

    fun start() {
        rotationSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    fun stop() = sensorManager.unregisterListener(this)

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_ROTATION_VECTOR) return
        SensorManager.getRotationMatrixFromVector(r9, event.values)
        // r9 is row-major device->world; copy to our Double Mat3.
        rotation = Mat3(DoubleArray(9) { r9[it].toDouble() })
        timestampNs = event.timestamp
        for (l in listeners) l()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
