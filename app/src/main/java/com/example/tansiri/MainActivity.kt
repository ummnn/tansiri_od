package com.example.tansiri

import android.Manifest
import android.app.Activity
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var cameraPermissionLauncher: ActivityResultLauncher<String>
    private lateinit var imageClassifier: ImageClassifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        cameraExecutor = Executors.newSingleThreadExecutor()
        imageClassifier = ImageClassifier(this)
        imageClassifier.initializeModel()

        requestCameraPermission()
    }

    private fun requestCameraPermission() {
        cameraPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                startCamera()
            }
        }
        cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable {
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val previewView = findViewById<PreviewView>(R.id.surfaceView)
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, ImageAnalysis.Analyzer { imageProxy ->
                        try {
                            // 이미지를 Bitmap으로 변환하고 모델에 전달
                            val bitmap = imageProxyToBitmap(imageProxy)
                            imageClassifier.classifyImage(bitmap)
                            bitmap.recycle() // 비트맵 해제
                        } finally {
                            imageProxy.close() // 사용한 이미지 프로세스 종료
                        }
                    })
                }

            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        // YUV -> RGB 변환 로직
        val planes = imageProxy.planes
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val yData = ByteArray(ySize)
        val uData = ByteArray(uSize)
        val vData = ByteArray(vSize)

        yBuffer.get(yData)
        uBuffer.get(uData)
        vBuffer.get(vData)

        // YUV -> RGB 변환
        val width = imageProxy.width
        val height = imageProxy.height
        val rgbArray = ByteArray(width * height * 4) // ARGB 8888 형식

        for (j in 0 until height) {
            for (i in 0 until width) {
                val yIndex = i + j * width
                val uIndex = (i / 2) + (j / 2) * (width / 2)
                val vIndex = (i / 2) + (j / 2) * (width / 2)

                val y = (yData[yIndex].toInt() and 0xFF).toFloat()
                val u = (uData[uIndex].toInt() and 0xFF).toFloat() - 128
                val v = (vData[vIndex].toInt() and 0xFF).toFloat() - 128

                var r = y + 1.402f * v
                var g = y - 0.344136f * u - 0.714136f * v
                var b = y + 1.772f * u

                r = r.coerceIn(0f, 255f)
                g = g.coerceIn(0f, 255f)
                b = b.coerceIn(0f, 255f)

                val rgbIndex = (i + j * width) * 4
                rgbArray[rgbIndex] = r.toInt().toByte() // R
                rgbArray[rgbIndex + 1] = g.toInt().toByte() // G
                rgbArray[rgbIndex + 2] = b.toInt().toByte() // B
                rgbArray[rgbIndex + 3] = 255.toByte() // Alpha (완전 불투명)
            }
        }

        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
            copyPixelsFromBuffer(ByteBuffer.wrap(rgbArray))
        }
    }
}

class ImageClassifier(private val activity: Activity) {
    private lateinit var interpreter: Interpreter
    private lateinit var probabilityBuffer: TensorBuffer
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var classNames: List<String>

    // 초기화 메서드
    fun initializeModel() {
        classNames = loadLabels() // label.txt에서 레이블 로드

        // 결과를 저장할 컨테이너 객체 생성
        probabilityBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 25200, 15), DataType.FLOAT32)

        // 이미지 프로세서 초기화
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(640, 640, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        // 모델 로드
        try {
            val tfliteModel: MappedByteBuffer = loadModelFile(activity, "third.tflite")
            interpreter = Interpreter(tfliteModel)
        } catch (e: IOException) {
            Log.e("tfliteSupport", "Error reading model", e)
        }
    }

    fun classifyImage(bitmap: Bitmap) {
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        val processedImage = imageProcessor.process(tensorImage)

        if (::interpreter.isInitialized) {
            try {
                interpreter.run(processedImage.buffer, probabilityBuffer.buffer)

                val results = probabilityBuffer.floatArray
                val detectedObjects = mutableListOf<String>()

                for (i in 0 until 25200) {
                    val confidence = results[i * 15 + 4] // 객체 확률 추출
                    if (confidence > 0.5f) { // 신뢰도 임계값 조정
                        // 각 클래스 확률 중 최대값 찾기
                        var maxClassProb = 0f
                        var maxClassIndex = -1
                        for (j in 5 until 15) { // 클래스 확률 영역 (5부터 15까지)
                            if (results[i * 15 + j] > maxClassProb) {
                                maxClassProb = results[i * 15 + j]
                                maxClassIndex = j - 5 // 클래스 인덱스는 0부터 시작하므로 5를 뺌
                            }
                        }

                        // 유효한 클래스인지 확인
                        if (maxClassIndex in classNames.indices) {
                            val detectedClass = "${classNames[maxClassIndex]}: ${confidence * maxClassProb * 100}%"
                            detectedObjects.add(detectedClass)
                        }
                    }
                }

            } catch (e: Exception) {
                Log.e("InferenceError", "Error during inference", e)
            }
        }
    }

    private fun loadLabels(): List<String> {
        val labels = mutableListOf<String>()
        try {
            activity.assets.open("label.txt").bufferedReader().useLines { lines ->
                lines.forEach { labels.add(it) }
            }

            // 레이블 로드 후 로그 출력
            Log.d("LabelLoad", "Loaded Labels: ${labels.joinToString(", ")}")

        } catch (e: IOException) {
            Log.e("LabelLoadError", "Error reading label file", e)
        }
        return labels
    }


    private fun loadModelFile(activity: Activity, modelFileName: String): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd(modelFileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength).also {
            inputStream.close() // 메모리 누수 방지를 위해 스트림 닫기
        }
    }
}
