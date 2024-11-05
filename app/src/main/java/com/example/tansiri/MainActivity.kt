package com.example.tansiri

import android.Manifest
import android.app.Activity
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.Bundle
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import android.util.Log
import android.view.View
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
    private lateinit var previewView: PreviewView
    private lateinit var boundingBoxOverlay: BoundingBoxOverlay

    // 하드코딩된 레이블 리스트
    private val associatedAxisLabels = listOf(
        "Car",
        "GTL",
        "Green PTL",
        "Green TL C",
        "Pedestrian Crossing C",
        "Pedestrian Crossing",
        "Person",
        "RTL",
        "Red PTL",
        "Red TL C"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        cameraExecutor = Executors.newSingleThreadExecutor()
        imageClassifier = ImageClassifier(this)
        imageClassifier.initializeModel()

        previewView = findViewById(R.id.surfaceView)
        boundingBoxOverlay = BoundingBoxOverlay(this)
        previewView.overlay.add(boundingBoxOverlay)  // boundingBoxOverlay를 overlay에 추가

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
                            imageClassifier.classifyImage(bitmap, associatedAxisLabels) { boundingBoxes ->
                                boundingBoxOverlay.updateBoundingBoxes(boundingBoxes)  // 바운딩 박스 업데이트
                            }
                        } finally {
                            imageProxy.close()
                        }
                    })
                }

            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        // YUV -> RGB 변환 로직 (이전 코드와 동일)
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

        val width = imageProxy.width
        val height = imageProxy.height
        val rgbArray = ByteArray(width * height * 4)

        for (j in 0 until height) {
            for (i in 0 until width) {
                val yIndex = i + j * width
                val uIndex = (i shr 1) + (j shr 1) * (width shr 1)
                val vIndex = (i shr 1) + (j shr 1) * (width shr 1)

                val y = (yData[yIndex].toInt() and 0xFF).toFloat()
                val u = (uData[uIndex].toInt() and 0xFF).toFloat() - 128
                val v = (vData[vIndex].toInt() and 0xFF).toFloat() - 128

                val r = (y + 1.402f * v).coerceIn(0f, 255f)
                val g = (y - 0.344136f * u - 0.714136f * v).coerceIn(0f, 255f)
                val b = (y + 1.772f * u).coerceIn(0f, 255f)

                val rgbIndex = (i + j * width) * 4
                rgbArray[rgbIndex] = r.toInt().toByte()
                rgbArray[rgbIndex + 1] = g.toInt().toByte()
                rgbArray[rgbIndex + 2] = b.toInt().toByte()
                rgbArray[rgbIndex + 3] = -1
            }
        }

        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
            copyPixelsFromBuffer(ByteBuffer.wrap(rgbArray))
        }
    }

    data class BoundingBox(val left: Float, val top: Float, val right: Float, val bottom: Float)

    inner class ImageClassifier(private val activity: Activity) {
        private lateinit var interpreter: Interpreter
        private lateinit var probabilityBuffer: TensorBuffer
        private lateinit var imageProcessor: ImageProcessor

        fun initializeModel() {
            probabilityBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 25200, 15), DataType.FLOAT32)

            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(640, 640, ResizeOp.ResizeMethod.BILINEAR))
                .build()

            try {
                val tfliteModel: MappedByteBuffer = loadModelFile(activity, "third.tflite")
                interpreter = Interpreter(tfliteModel)
            } catch (e: IOException) {
                Log.e("tfliteSupport", "Error reading model", e)
            }
        }

        fun classifyImage(
            bitmap: Bitmap,
            associatedAxisLabels: List<String>,
            onBoundingBoxesDetected: (List<BoundingBox>) -> Unit
        ) {
            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)

            if (::interpreter.isInitialized) {
                interpreter.run(processedImage.buffer, probabilityBuffer.buffer)

                val results = probabilityBuffer.floatArray
                val boundingBoxes = mutableListOf<BoundingBox>()
                for (i in results.indices step 15) {
                    val left = results[i + 1] * bitmap.width
                    val top = results[i + 2] * bitmap.height
                    val right = results[i + 3] * bitmap.width
                    val bottom = results[i + 4] * bitmap.height

                    boundingBoxes.add(BoundingBox(left, top, right, bottom))
                }
                onBoundingBoxesDetected(boundingBoxes)
            }
        }

        private fun loadModelFile(activity: Activity, modelName: String): MappedByteBuffer {
            val fileDescriptor = activity.assets.openFd(modelName)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }
    }

    inner class BoundingBoxOverlay(context: Activity) : View(context) {
        private val paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 8f
        }
        private var boundingBoxes = listOf<BoundingBox>()

        fun updateBoundingBoxes(newBoxes: List<BoundingBox>) {
            boundingBoxes = newBoxes
            invalidate() // 화면 갱신
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            for (box in boundingBoxes) {
                val rect = RectF(box.left, box.top, box.right, box.bottom)
                canvas.drawRect(rect, paint)
            }
        }
    }
}
