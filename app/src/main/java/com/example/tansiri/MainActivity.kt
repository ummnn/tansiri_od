package com.example.tansiri

import android.Manifest
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.os.Bundle
import android.util.Log
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var cameraPermissionLauncher: ActivityResultLauncher<String>
    private var lastAnalyzedTime = 0L // 타임스탬프 초기화
    private lateinit var interpreter: Interpreter

    companion object {
        private const val INPUT_WIDTH = 640 // 모델의 입력 너비
        private const val INPUT_HEIGHT = 640 // 모델의 입력 높이
        private const val NUM_CLASSES = 10 // 모델의 클래스 수
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 카메라를 위한 Executor 초기화
        cameraExecutor = Executors.newSingleThreadExecutor()

        // TFLite 모델 초기화
        interpreter = Interpreter(loadModelFile(applicationContext, "model.tflite")) // TFLite 모델 파일 경로

        // 카메라 권한 요청
        requestCameraPermission()
    }

    // 카메라 권한 요청
    private fun requestCameraPermission() {
        cameraPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                startCamera()
            } else {
                // 권한이 거부된 경우 처리
            }
        }
        cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
    }

    // 카메라 시작
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable {
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // 프리뷰 설정
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(findViewById<PreviewView>(R.id.previewView).surfaceProvider)
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            // 이미지 분석을 위한 ImageAnalysis 설정
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, ImageAnalysis.Analyzer { imageProxy ->
                        // 이미지 분석: 이미지를 TFLiteModel에 전달
                        sendImageToModel(imageProxy)
                    })
                }

            // 카메라 라이프사이클에 바인딩
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
        }, ContextCompat.getMainExecutor(this))
    }

    // TFLite 모델에 이미지 데이터 전달
    private fun sendImageToModel(imageProxy: ImageProxy) {
        val currentTime = System.currentTimeMillis()
        // 500ms마다 한 번씩 추론 (초당 약 2 프레임)
        if (currentTime - lastAnalyzedTime >= 500) {
            lastAnalyzedTime = currentTime

            // YUV 플레인 데이터 가져오기
            val yBuffer = imageProxy.planes[0].buffer // Y
            val uBuffer = imageProxy.planes[1].buffer // U
            val vBuffer = imageProxy.planes[2].buffer // V

            val yData = ByteArray(yBuffer.remaining())
            val uData = ByteArray(uBuffer.remaining())
            val vData = ByteArray(vBuffer.remaining())

            yBuffer.get(yData)
            uBuffer.get(uData)
            vBuffer.get(vData)

            val width = imageProxy.width
            val height = imageProxy.height

            Log.d("YUVData", "Y size: ${yData.size}, U size: ${uData.size}, V size: ${vData.size}")

            try {
                // TFLite 모델에 YUV 데이터를 전달하여 추론 수행
                val output = runInference(yData, uData, vData, width, height)

                // 추론 결과를 로그에 출력
                Log.d("InferenceResult", output.contentToString())
            } catch (e: Exception) {
                Log.e("InferenceError", "Error during inference: ${e.message}")
            }
        }
        // 이미지 프록시 닫기
        imageProxy.close()
    }

    // TFLite 모델에 대한 추론 수행
    private fun runInference(yData: ByteArray, uData: ByteArray, vData: ByteArray, width: Int, height: Int): Array<FloatArray> {
        // YUV 데이터를 Bitmap으로 변환
        val bitmap = convertYUVtoBitmap(yData, uData, vData, width, height)

        // 비트맵을 모델 입력 크기로 리사이즈
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_WIDTH, INPUT_HEIGHT, true)

        // 리사이즈된 비트맵을 ByteBuffer로 변환
        val inputBuffer = bitmapToByteBuffer(resizedBitmap)

        // 추론 결과를 위한 배열
        val output = Array(1) { FloatArray(NUM_CLASSES) } // NUM_CLASSES: 모델 클래스 수

        // TFLite 모델로 추론 호출
        interpreter.run(inputBuffer, output)

        return output
    }

    // YUV 데이터를 Bitmap으로 변환
    private fun convertYUVtoBitmap(yData: ByteArray, uData: ByteArray, vData: ByteArray, width: Int, height: Int): Bitmap {
        // YUV 데이터를 NV21 형식으로 결합
        val nv21 = ByteArray(yData.size + uData.size + vData.size)
        System.arraycopy(yData, 0, nv21, 0, yData.size)
        System.arraycopy(uData, 0, nv21, yData.size, uData.size)
        System.arraycopy(vData, 0, nv21, yData.size + uData.size, vData.size)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }


// 비트맵을 ByteBuffer로 변환
    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(INPUT_WIDTH * INPUT_HEIGHT * 3)
        buffer.order(ByteOrder.nativeOrder())

        for (y in 0 until INPUT_HEIGHT) {
            for (x in 0 until INPUT_WIDTH) {
                val pixel = bitmap.getPixel(x, y)

                // RGB 값을 INT8 형식으로 변환 (-128~127 범위로 변환)
                val r = (Color.red(pixel) - 128).toByte() // 0-255 -> -128~127
                val g = (Color.green(pixel) - 128).toByte()
                val b = (Color.blue(pixel) - 128).toByte()

                // ByteBuffer에 값 추가
                buffer.put(r)
                buffer.put(g)
                buffer.put(b)
            }
        }

        buffer.rewind() // 버퍼의 포인터를 초기 위치로 재설정
        return buffer
    }






    // 모델 파일 로드
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    override fun onDestroy() {
        super.onDestroy()
        // Executor 종료
        cameraExecutor.shutdown()
        // TFLite 모델 자원 해제
        interpreter.close() // 인터프리터 종료
    }
}
