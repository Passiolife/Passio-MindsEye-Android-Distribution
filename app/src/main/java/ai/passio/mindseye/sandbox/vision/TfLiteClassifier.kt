package ai.passio.mindseye.sandbox.vision

import android.content.Context
import android.graphics.Bitmap
import android.util.Size
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.tensorflow.lite.support.metadata.schema.NormalizationOptions
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class TfLiteClassifier(
    context: Context,
    fileName: String
) {

    companion object {
        private val NUM_CORES: Int by lazy {
            var availableCores = Runtime.getRuntime().availableProcessors()
            if (availableCores > 1) {
                availableCores /= 2
            }
            availableCores
        }

        private const val BYTES_PER_CHANNEL = 4
        private const val IS_QUANTIZED = false
    }

    private var std = -1f
    private var mean = -1f

    lateinit var inputSize: Size
    private lateinit var imgData: ByteBuffer
    private lateinit var intValues: IntArray

    private val tfLite: Interpreter

    private lateinit var labels: List<String>
    private val results: Array<FloatArray> by lazy {
        init2DFloatArray(1, labels.size)
    }

    init {
        val assetFileDescriptor = context.assets.openFd(fileName)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        val buffer = inputStream.channel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

        readMetadata(buffer)
        tfLite = Interpreter(buffer, getDefaultTFLiteOptions())
    }

    private fun getDefaultTFLiteOptions(): Interpreter.Options {
        return Interpreter.Options().apply {
            numThreads = NUM_CORES
            setUseXNNPACK(true)
        }
    }

    private fun readMetadata(modelBuffer: ByteBuffer) {
        val extractor = MetadataExtractor(modelBuffer)

        if (!extractor.hasMetadata() || !extractor.isMinimumParserVersionSatisfied) {
            throw java.lang.RuntimeException("Not a viable tflite model, could not read metadata!")
        }

        val inputTensorShape = extractor.getInputTensorShape(0)!!
        inputSize = Size(inputTensorShape[1], inputTensorShape[2])

        val normalizationOptions = extractor.getInputTensorMetadata(0)?.processUnits(0)
            ?.options(NormalizationOptions()) as? NormalizationOptions
        if (normalizationOptions != null) {
            std = normalizationOptions.std(0)
            mean = normalizationOptions.mean(0)
        } else {
            throw java.lang.RuntimeException("Not a viable tflite model, could not read metadata std and mean!")
        }

        imgData = ByteBuffer.allocateDirect(
            1 * inputSize.width * inputSize.height * 3 * BYTES_PER_CHANNEL
        ).apply { order(ByteOrder.nativeOrder()) }

        intValues = IntArray(inputSize.width * inputSize.height)

        val labelFilename = extractor.associatedFileNames.toList().first()
        val inputStream = extractor.getAssociatedFile(labelFilename)
        val labelsTemp = mutableListOf<String>()
        inputStream.bufferedReader().useLines { lines ->
            lines.forEach { line ->
                labelsTemp.add(line)
            }
        }
        labels = labelsTemp
    }

    fun recognizeImage(bitmap: Bitmap): ClassificationCandidate {
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        imgData.rewind()
        for (i in 0 until inputSize.width) {
            for (j in 0 until inputSize.height) {
                val pixelValue = intValues[i * inputSize.width + j]
                if (IS_QUANTIZED) {
                    imgData.put((pixelValue shr 16 and 0xFF).toByte())
                    imgData.put((pixelValue shr 8 and 0xFF).toByte())
                    imgData.put((pixelValue and 0xFF).toByte())
                } else {
                    imgData.putFloat(((pixelValue shr 16 and 0xFF) - mean) / std)
                    imgData.putFloat(((pixelValue shr 8 and 0xFF) - mean) / std)
                    imgData.putFloat(((pixelValue and 0xFF) - mean) / std)
                }
            }
        }

        val inputArray: Array<Any> = arrayOf(imgData)
        val outputMap = mapOf(
            0 to results,
        )

        tfLite.runForMultipleInputsOutputs(inputArray, outputMap)

        val predictedProbs = outputMap.getValue(0)[0]

        var maxValue = 0f
        var maxIndex = 0
        for (i in predictedProbs.indices) {
            if (predictedProbs[i] > maxValue) {
                maxValue = predictedProbs[i]
                maxIndex = i
            }
        }

        val label = labels[maxIndex]
        return ClassificationCandidate(label, maxValue)
    }

    fun init2DFloatArray(
        dim1: Int, dim2: Int,
        init: (Int, Int) -> Float = { _, _ -> 0f }
    ): Array<FloatArray> {
        return Array(dim1) { i ->
            FloatArray(dim2) { j ->
                init(i, j)
            }
        }
    }
}