package ai.passio.mindseye.sandbox

import ai.passio.mindseye.sandbox.databinding.ActivityMainBinding
import ai.passio.mindseye.sandbox.vision.*
import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.util.*

private const val MODE_FILENAME = "model.tflite"

class MainActivity : AppCompatActivity(), ClassificationListener {

    private lateinit var classifier: TfLiteClassifier
    private lateinit var cameraController: CameraController
    private lateinit var recognizer: MindsEyeRecognizer

    private val requestPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted: Boolean ->
            if (isGranted) {
                onCameraPermissionGranted()
            } else {
                onCameraPermissionDenied()
            }
        }

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        classifier = TfLiteClassifier(this, MODE_FILENAME)
        recognizer = MindsEyeRecognizer(this.applicationContext, classifier, this)
        cameraController = CameraController(recognizer).apply {
            enableTapToFocus(binding.mainPreview)
        }

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            onCameraPermissionGranted()
        } else {
            // You can directly ask for the permission.
            // The registered ActivityResultCallback gets the result of this request.
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    @SuppressLint("SetTextI18n")
    override fun onClassificationCandidate(candidate: ClassificationCandidate) {
        binding.mainResult.text = "${candidate.label.cap()} (${String.format("%.2f", candidate.confidence)})"
    }

    private fun onCameraPermissionGranted() {
        cameraController.startCamera(binding.mainPreview, this)
    }

    private fun onCameraPermissionDenied() {
        Toast.makeText(
            this,
            "This app requires camera to work!",
            Toast.LENGTH_LONG
        ).show()
    }

    private fun String.cap(): String {
        return this.replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.getDefault()) else it.toString() }
    }

    private fun loadBitmapFromAssets(assetManager: AssetManager, imageName: String): Bitmap {
        val inputStream = assetManager.open(imageName)
        return BitmapFactory.decodeStream(inputStream)
    }
}