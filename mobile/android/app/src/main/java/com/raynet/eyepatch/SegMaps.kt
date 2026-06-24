package com.raynet.eyepatch

/**
 * Three segmentation probability planes (row-major, length w*h), one per eye
 * feature, as emitted by [EyeSegmenter] and consumed by [SegPostProcess].
 * Kept in its own (Android-free) file so the tier-2 logic can be JVM-tested.
 */
class SegMaps(
    val pupil: FloatArray,
    val iris: FloatArray,
    val sclera: FloatArray,
    val w: Int,
    val h: Int,
)
