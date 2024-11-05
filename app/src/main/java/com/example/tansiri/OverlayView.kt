package com.example.tansiri

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Log
import android.view.View

class OverlayView(context: Context, attrs: AttributeSet) : View(context, attrs) {
    private val paint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val boundingBoxes = mutableListOf<RectF>()

    fun setBoundingBoxes(boxes: List<RectF>) {
        boundingBoxes.clear()
        boundingBoxes.addAll(boxes)
        invalidate()  // Refresh the view to redraw bounding boxes
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        Log.d("OverlayView", "Drawing ${boundingBoxes.size} bounding boxes")
        for (box in boundingBoxes) {
            canvas.drawRect(box, paint)
        }
    }

}
