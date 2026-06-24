plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.raynet.eyepatch"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.raynet.eyepatch"
        minSdk = 24            // MediaPipe tasks-vision requires 24+
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    // Do NOT compress the model assets — MediaPipe mmaps the .task/.tflite, and
    // keeping the .ort uncompressed avoids a copy on load.
    androidResources {
        noCompress += listOf("ort", "task", "tflite")
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.2")

    // CameraX (toBitmap() needs >= 1.3.0).
    val camerax = "1.3.4"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")

    // ONNX Runtime Mobile (full op build; loads the .ort flatbuffer). Used for
    // the head-pose net (RepNeXt-M4).
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")

    // TensorFlow Lite + GPU delegate for the segmentation net (real-time on the
    // Adreno GPU; the ONNX CPU path was the frame-rate wall).
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.16.1")

    // MediaPipe Face Landmarker (478 landmarks incl. iris).
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
}
