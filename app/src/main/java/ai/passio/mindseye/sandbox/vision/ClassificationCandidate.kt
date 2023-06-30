package ai.passio.mindseye.sandbox.vision

data class ClassificationCandidate(
    val label: String,
    val confidence: Float
)

interface ClassificationListener {
    fun onClassificationCandidate(candidate: ClassificationCandidate)
}