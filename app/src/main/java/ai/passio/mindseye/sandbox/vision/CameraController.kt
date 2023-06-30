package ai.passio.mindseye.sandbox.vision

import android.annotation.SuppressLint
import android.util.Log
import android.util.Size
import android.view.MotionEvent
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraController(
    private val cameraListener: CameraListener,
    private val frameTime: Long = 500L
) {

    interface CameraListener {
        fun onFrameSize(
            frameWidth: Int,
            frameHeight: Int,
            previewWidth: Int,
            previewHeight: Int,
            orientation: Int
        )

        fun analyzeImage(imageProxy: ImageProxy)
    }

    private var camera: Camera? = null
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null
    private var frameSize: Size? = null
    private var previewSize: Size? = null

    @SuppressLint("RestrictedApi")
    fun startCamera(
        previewView: PreviewView,
        lifecycleOwner: LifecycleOwner,
        displayRotation: Int = 0,
        lensFacing: Int = CameraSelector.LENS_FACING_BACK,
        onCameraReady: () -> Unit = {}
    ) {
        val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(previewView.context)
        cameraProviderFuture.addListener({

            cameraProvider = cameraProviderFuture.get() as ProcessCameraProvider
            cameraProvider!!.unbindAll()

            val preview = Preview.Builder().apply {
                setTargetRotation(displayRotation)
                setTargetAspectRatio(AspectRatio.RATIO_16_9)
            }.build().also {
                previewView.implementationMode =
                    PreviewView.ImplementationMode.PERFORMANCE
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder().apply {
                setTargetRotation(displayRotation)
                setTargetAspectRatio(AspectRatio.RATIO_16_9)
            }.build().also {
                it.setAnalyzer(cameraExecutor, CameraAnalyzer())
            }

            try {
                camera =
                    cameraProvider!!.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview,
                        imageAnalyzer
                    )
                frameSize = imageAnalyzer.attachedSurfaceResolution
                previewSize = preview.attachedSurfaceResolution

                cameraListener.onFrameSize(
                    frameSize!!.width,
                    frameSize!!.height,
                    previewSize!!.width,
                    previewSize!!.height,
                    camera!!.cameraInfo.sensorRotationDegrees
                )

                Log.i(
                    this::class.java.simpleName,
                    "Frame size: $frameSize, preview size: $previewSize"
                )
                onCameraReady()
            } catch (e: Exception) {
                Log.e(this::class.java.simpleName, "Use case binding failed", e)
            }

        }, ContextCompat.getMainExecutor(previewView.context))
    }

    @SuppressLint("ClickableViewAccessibility")
    fun enableTapToFocus(
        previewView: PreviewView
    ) {
        previewView.setOnTouchListener { v, event ->
            return@setOnTouchListener when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    true
                }
                MotionEvent.ACTION_UP -> {
                    val factory: MeteringPointFactory = SurfaceOrientedMeteringPointFactory(
                        previewView.width.toFloat(), previewView.height.toFloat()
                    )
                    val autoFocusPoint = factory.createPoint(event.x, event.y)
                    try {
                        camera?.cameraControl?.startFocusAndMetering(
                            FocusMeteringAction.Builder(
                                autoFocusPoint,
                                FocusMeteringAction.FLAG_AF
                            ).disableAutoCancel().build()
                        )
                    } catch (e: CameraInfoUnavailableException) {
                        Log.e(
                            this::class.java.simpleName,
                            "Cannot access camera, manual focus failed",
                            e
                        )
                    }
                    true
                }
                else -> false
            }
        }
    }

    private inner class CameraAnalyzer : ImageAnalysis.Analyzer {

        private var lastAnalyzedTimestamp = 0L

        @SuppressLint("UnsafeOptInUsageError")
        override fun analyze(image: ImageProxy) {
            if (image.image == null) {
                image.close()
                return
            }

            val currentTimestamp = System.currentTimeMillis()
            if (currentTimestamp - lastAnalyzedTimestamp >= frameTime) {
                lastAnalyzedTimestamp = currentTimestamp
                cameraListener.analyzeImage(image)
            } else {
                image.close()
            }
        }
    }
}