package ai.passio.mindseye.sandbox.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import androidx.core.content.ContextCompat
import kotlin.math.abs
import kotlin.math.max

private const val kMaxChannelValue = 262143

class MindsEyeRecognizer(
    context: Context,
    private val classifier: TfLiteClassifier,
    private val listener: ClassificationListener,
) : CameraController.CameraListener {

    private val mainExecutor = ContextCompat.getMainExecutor(context)

    private var frameWidth = -1
    private var frameHeight = -1
    private var orientation = -1
    private var rgbBytes: IntArray? = null
    private val yuvBytes = init2DByteArray(3, null)

    private val croppedClassBitmap: Bitmap by lazy {
        Bitmap.createBitmap(
            classifier.inputSize.width,
            classifier.inputSize.height,
            Bitmap.Config.ARGB_8888
        )
    }

    private val frameToClassTransform: Matrix by lazy {
        getTransformationMatrix(
            frameWidth,
            frameHeight,
            classifier.inputSize.width,
            classifier.inputSize.height,
            orientation,
            false
        )
    }

    fun recognizeImage(
        bitmap: Bitmap,
    ): ClassificationCandidate {
        val scaled = Bitmap.createScaledBitmap(bitmap, classifier.inputSize.width, classifier.inputSize.width, true)
        val candidate = classifier.recognizeImage(scaled)
        scaled.recycle()
        return candidate
    }

    override fun onFrameSize(
        frameWidth: Int,
        frameHeight: Int,
        previewWidth: Int,
        previewHeight: Int,
        orientation: Int
    ) {
        this.frameWidth = frameWidth
        this.frameHeight = frameHeight
        this.orientation = orientation
    }

    override fun analyzeImage(imageProxy: ImageProxy) {
        if (imageProxy.width != frameWidth && imageProxy.height != frameHeight) {
            imageProxy.close()
            return
        }

        rgbBytes = IntArray(imageProxy.width * imageProxy.height)

        val planes = imageProxy.planes
        imageProxyToYUV(planes, yuvBytes)

        val yRowStride = planes[0].rowStride
        val uvRowStride = planes[1].rowStride
        val uvPixelStride = planes[1].pixelStride

        convertYUV420ToARGB8888(
            yuvBytes[0]!!,
            yuvBytes[1]!!,
            yuvBytes[2]!!,
            imageProxy.width,
            imageProxy.height,
            yRowStride,
            uvRowStride,
            uvPixelStride,
            rgbBytes!!
        )

        val bitmap = Bitmap.createBitmap(
            rgbBytes!!,
            imageProxy.width,
            imageProxy.height,
            Bitmap.Config.ARGB_8888
        )

        val classCanvas = Canvas(croppedClassBitmap)
        classCanvas.drawBitmap(bitmap, frameToClassTransform, null)
        val candidate = classifier.recognizeImage(croppedClassBitmap)
        bitmap.recycle()

        mainExecutor.execute {
            listener.onClassificationCandidate(candidate)
        }

        imageProxy.close()
    }

    private fun convertYUV420ToARGB8888(
        yData: ByteArray,
        uData: ByteArray,
        vData: ByteArray,
        width: Int,
        height: Int,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        out: IntArray
    ) {
        var yp = 0
        for (j in 0 until height) {
            val pY = yRowStride * j
            val pUV = uvRowStride * (j shr 1)

            for (i in 0 until width) {
                val uv_offset = pUV + (i shr 1) * uvPixelStride

                try {
                    out[yp++] = yuv2rgb(
                        0xff and yData[pY + i].toInt(),
                        0xff and uData[uv_offset].toInt(),
                        0xff and vData[uv_offset].toInt()
                    )
                } catch (e: IndexOutOfBoundsException) {
                    e.printStackTrace()
                    throw e
                }

            }
        }
    }

    private fun yuv2rgb(yValue: Int, uValue: Int, vValue: Int): Int {
        var y = yValue
        var u = uValue
        var v = vValue
        // Adjust and check YUV values
        y = if (y - 16 < 0) 0 else y - 16
        u -= 128
        v -= 128

        // This is the floating point equivalent. We do the conversion in integer
        // because some Android devices do not have floating point in hardware.
        // nR = (int)(1.164 * nY + 2.018 * nU);
        // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
        // nB = (int)(1.164 * nY + 1.596 * nV);
        val y1192 = 1192 * y
        var r = y1192 + 1634 * v
        var g = y1192 - 833 * v - 400 * u
        var b = y1192 + 2066 * u

        // Clipping RGB values to be inside boundaries [ 0 , kMaxChannelValue ]
        r = if (r > kMaxChannelValue) kMaxChannelValue else if (r < 0) 0 else r
        g = if (g > kMaxChannelValue) kMaxChannelValue else if (g < 0) 0 else g
        b = if (b > kMaxChannelValue) kMaxChannelValue else if (b < 0) 0 else b

        return -0x1000000 or (r shl 6 and 0xff0000) or (g shr 2 and 0xff00) or (b shr 10 and 0xff)
    }

    private fun init2DByteArray(
        dim1: Int, dim2: Int?,
        init: (Int, Int) -> Byte = { _, _ -> 0 }
    ): Array<ByteArray?> {
        return Array(dim1) { i ->
            if (dim2 == null) {
                null
            } else {
                ByteArray(dim2) { j ->
                    init(i, j)
                }
            }
        }
    }

    private fun imageProxyToYUV(
        planes: Array<ImageProxy.PlaneProxy>,
        yuvBytes: Array<ByteArray?>
    ) {
        planes.indices.forEach { i ->
            val buffer = planes[i].buffer
            buffer.rewind()
            if (yuvBytes[i] == null || yuvBytes[i]!!.size != buffer.capacity()) {
                yuvBytes[i] = ByteArray(buffer.capacity())
            }
            buffer.get(yuvBytes[i]!!)
        }
    }

    private fun getTransformationMatrix(
        srcWidth: Int,
        srcHeight: Int,
        dstWidth: Int,
        dstHeight: Int,
        applyRotation: Int,
        maintainAspectRatio: Boolean
    ): Matrix {
        val matrix = Matrix()

        if (applyRotation != 0) {
            if (applyRotation % 90 != 0) {
                // Don't apply non-standard rotation
            } else {
                matrix.postTranslate(-srcWidth / 2f, -srcHeight / 2f)
                matrix.postRotate(applyRotation.toFloat())
            }
        }

        val transpose = (abs(applyRotation) + 90) % 180 == 0

        val inWidth = if (transpose) srcHeight else srcWidth
        val inHeight = if (transpose) srcWidth else srcHeight

        if (inWidth != dstWidth || inHeight != dstHeight) {
            val scaleFactorX = dstWidth / inWidth.toFloat()
            val scaleFactorY = dstHeight / inHeight.toFloat()

            if (maintainAspectRatio) {
                val scaleFactor = max(scaleFactorX, scaleFactorY)
                matrix.postScale(scaleFactor, scaleFactor)
            } else {
                matrix.postScale(scaleFactorX, scaleFactorY)
            }
        }

        if (applyRotation != 0) {
            matrix.postTranslate(dstWidth / 2f, dstHeight / 2f)
        }

        return matrix
    }
}