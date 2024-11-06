import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

data class BoundingBox(val left: Float, val top: Float, val right: Float, val bottom: Float)

class BoundingBoxOverlay @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {
    private val paint = Paint().apply {
        color = 0xFFFF0000.toInt() // 빨간색
        strokeWidth = 8f
        style = Paint.Style.STROKE
    }

    private var boundingBoxes: List<BoundingBox> = emptyList()

    fun setBoundingBoxes(boxes: List<BoundingBox>) {
        this.boundingBoxes = boxes
        invalidate() // 뷰를 다시 그리기
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        for (box in boundingBoxes) {
            canvas.drawRect(box.left, box.top, box.right, box.bottom, paint)
        }
    }
}
