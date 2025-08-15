import os
import asyncio
import json
import cv2
import numpy as np
from typing import Optional, Dict, Any, List
import base64
from datetime import datetime
from io import BytesIO
from PIL import Image
import tensorflow as tf

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Extension Box Detection API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model interpreter
interpreter = None
input_details = None
output_details = None

class DetectionResult(BaseModel):
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_name: str

class DetectionResponse(BaseModel):
    detections: List[DetectionResult]
    image_width: int
    image_height: int
    processing_time: float

class ImageRequest(BaseModel):
    image_base64: str
    image_width: int
    image_height: int

def load_model():
    """Load the TensorFlow Lite model"""
    global interpreter, input_details, output_details
    
    try:
        model_path = "/app/models/best-fp16.tflite"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Model loaded successfully!")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Output details: {len(output_details)} outputs")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_image(image: np.ndarray, target_size: tuple = (640, 640)) -> np.ndarray:
    """Preprocess image for model inference"""
    # Resize image while maintaining aspect ratio
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    padded = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    y_offset = (target_size[1] - new_h) // 2
    x_offset = (target_size[0] - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    # Normalize to [0, 1] if model expects float input
    if input_details[0]['dtype'] == np.float32:
        padded = padded.astype(np.float32) / 255.0
    
    # Add batch dimension
    input_data = np.expand_dims(padded, axis=0)
    
    return input_data, scale, x_offset, y_offset

def postprocess_detections(outputs: List[np.ndarray], scale: float, x_offset: int, y_offset: int, 
                         original_width: int, original_height: int, conf_threshold: float = 0.5) -> List[DetectionResult]:
    """Process model outputs to get detection results"""
    try:
        detections = []
        
        # Handle different output formats (YOLO-style)
        if len(outputs) >= 1:
            # Assuming YOLO format: [batch, num_detections, 85] where 85 = 4 (box) + 1 (conf) + 80 (classes)
            output = outputs[0][0]  # Remove batch dimension
            
            boxes = []
            confidences = []
            
            for detection in output:
                if len(detection) >= 5:  # At least x, y, w, h, confidence
                    confidence = detection[4]
                    
                    if confidence > conf_threshold:
                        # Extract box coordinates (center_x, center_y, width, height)
                        center_x, center_y, width, height = detection[:4]
                        
                        # Convert to corner coordinates
                        x1 = center_x - width / 2
                        y1 = center_y - height / 2
                        x2 = center_x + width / 2
                        y2 = center_y + height / 2
                        
                        # Adjust for preprocessing transformations
                        x1 = (x1 - x_offset) / scale
                        y1 = (y1 - y_offset) / scale
                        x2 = (x2 - x_offset) / scale
                        y2 = (y2 - y_offset) / scale
                        
                        # Clamp to image boundaries
                        x1 = max(0, min(x1, original_width))
                        y1 = max(0, min(y1, original_height))
                        x2 = max(0, min(x2, original_width))
                        y2 = max(0, min(y2, original_height))
                        
                        # Only add if box is valid
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            confidences.append(float(confidence))
            
            # Apply NMS to remove overlapping boxes
            if boxes:
                indices = cv2.dnn.NMSBoxes(
                    [(x1, y1, x2-x1, y2-y1) for x1, y1, x2, y2 in boxes],
                    confidences,
                    conf_threshold,
                    0.4  # NMS threshold
                )
                
                if len(indices) > 0:
                    # Get most confident detection
                    best_idx = np.argmax([confidences[i] for i in indices.flatten()])
                    idx = indices.flatten()[best_idx]
                    
                    x1, y1, x2, y2 = boxes[idx]
                    confidence = confidences[idx]
                    
                    detections.append(DetectionResult(
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        confidence=float(confidence),
                        class_name="extension_box"
                    ))
        
        return detections
    
    except Exception as e:
        print(f"Error in postprocessing: {e}")
        return []

async def run_inference(image: np.ndarray, original_width: int, original_height: int) -> List[DetectionResult]:
    """Run inference on the image"""
    global interpreter, input_details, output_details
    
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        start_time = datetime.now()
        
        # Preprocess image
        target_size = tuple(input_details[0]['shape'][1:3])  # Get height, width from model
        input_data, scale, x_offset, y_offset = preprocess_image(image, target_size)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get outputs
        outputs = []
        for output_detail in output_details:
            outputs.append(interpreter.get_tensor(output_detail['index']))
        
        # Postprocess
        detections = postprocess_detections(
            outputs, scale, x_offset, y_offset, 
            original_width, original_height
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"Inference completed in {processing_time:.3f}s, found {len(detections)} detections")
        
        return detections
    
    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        print("WARNING: Model failed to load!")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if interpreter is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/detect")
async def detect_objects(request: ImageRequest) -> DetectionResponse:
    """Detect objects in the provided image"""
    try:
        start_time = datetime.now()
        
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64.split(',')[-1])
        image = Image.open(BytesIO(image_data))
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run inference
        detections = await run_inference(cv_image, request.image_width, request.image_height)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DetectionResponse(
            detections=detections,
            image_width=request.image_width,
            image_height=request.image_height,
            processing_time=processing_time
        )
    
    except Exception as e:
        print(f"Error in detect endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket manager for real-time detection
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))

manager = ConnectionManager()

@app.websocket("/api/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """WebSocket endpoint for real-time detection"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive image data
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "image":
                try:
                    # Process the image
                    image_base64 = message["image"]
                    width = message["width"]
                    height = message["height"]
                    
                    # Decode and process image
                    image_data = base64.b64decode(image_base64.split(',')[-1])
                    image = Image.open(BytesIO(image_data))
                    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Run inference
                    detections = await run_inference(cv_image, width, height)
                    
                    # Send results back
                    response = {
                        "type": "detection_result",
                        "detections": [
                            {
                                "bbox": det.bbox,
                                "confidence": det.confidence,
                                "class_name": det.class_name
                            } for det in detections
                        ],
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    await manager.send_personal_message(response, websocket)
                
                except Exception as e:
                    error_response = {
                        "type": "error",
                        "message": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    await manager.send_personal_message(error_response, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket client disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)