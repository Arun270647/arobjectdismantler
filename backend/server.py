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
                         original_width: int, original_height: int, conf_threshold: float = 0.1) -> List[DetectionResult]:
    """Process model outputs to get detection results"""
    try:
        detections = []
        
        if len(outputs) >= 1:
            # YOLO format: [batch, num_detections, 85] where 85 = 4 (box) + 1 (objectness) + 80 (classes)
            output = outputs[0][0]  # Remove batch dimension, shape: (25200, 85)
            
            boxes = []
            confidences = []
            class_ids = []
            
            # Debug: check max values
            max_objectness = np.max(output[:, 4])
            max_class_score = np.max(output[:, 5:])
            print(f"Debug - Max objectness: {max_objectness:.6f}, Max class score: {max_class_score:.6f}")
            
            # Count detections above different thresholds
            count_01 = np.sum(output[:, 4] > 0.01)
            count_05 = np.sum(output[:, 4] > 0.05) 
            count_1 = np.sum(output[:, 4] > 0.1)
            count_3 = np.sum(output[:, 4] > 0.3)
            print(f"Debug - Detections above thresholds: 0.01:{count_01}, 0.05:{count_05}, 0.1:{count_1}, 0.3:{count_3}")
            
            for i, detection in enumerate(output):
                # detection[4] is objectness score
                objectness = detection[4]
                
                if objectness > conf_threshold:
                    # Get class probabilities (indices 5-84)
                    class_scores = detection[5:]
                    class_id = np.argmax(class_scores)
                    class_confidence = class_scores[class_id]
                    
                    # For extension box detection, use objectness as main confidence
                    # since class confidence might be less reliable for specific object types
                    confidence = objectness
                    
                    if confidence > conf_threshold:
                        # Extract box coordinates (normalized to 0-640)
                        center_x, center_y, width, height = detection[:4]
                        
                        # Convert to corner coordinates (still in model coordinate system)
                        x1 = center_x - width / 2
                        y1 = center_y - height / 2
                        x2 = center_x + width / 2
                        y2 = center_y + height / 2
                        
                        # Transform back to original image coordinates
                        # First, convert from model coords (0-640) to padded image coords
                        x1_padded = x1
                        y1_padded = y1
                        x2_padded = x2
                        y2_padded = y2
                        
                        # Then remove padding offset
                        x1_resized = x1_padded - x_offset
                        y1_resized = y1_padded - y_offset
                        x2_resized = x2_padded - x_offset
                        y2_resized = y2_padded - y_offset
                        
                        # Finally scale back to original image size
                        x1_orig = x1_resized / scale
                        y1_orig = y1_resized / scale
                        x2_orig = x2_resized / scale
                        y2_orig = y2_resized / scale
                        
                        # Clamp to image boundaries
                        x1_orig = max(0, min(x1_orig, original_width))
                        y1_orig = max(0, min(y1_orig, original_height))
                        x2_orig = max(0, min(x2_orig, original_width))
                        y2_orig = max(0, min(y2_orig, original_height))
                        
                        # Only add if box is valid
                        if x2_orig > x1_orig and y2_orig > y1_orig:
                            boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
                            confidences.append(float(confidence))
                            class_ids.append(int(class_id))
                            
                            print(f"Debug - Detection {len(boxes)}: obj={objectness:.4f}, cls={class_confidence:.4f}, conf={confidence:.4f}, class_id={class_id}")
            
            print(f"Found {len(boxes)} potential detections before NMS")
            
            # Apply NMS to remove overlapping boxes
            if boxes and len(boxes) > 0:
                # Convert to format expected by cv2.dnn.NMSBoxes
                nms_boxes = [(x1, y1, x2-x1, y2-y1) for x1, y1, x2, y2 in boxes]
                
                indices = cv2.dnn.NMSBoxes(
                    nms_boxes,
                    confidences,
                    conf_threshold,
                    0.4  # NMS threshold
                )
                
                print(f"After NMS: {len(indices) if len(indices) > 0 else 0} detections")
                
                if len(indices) > 0:
                    # Get most confident detection (as requested)
                    if isinstance(indices, np.ndarray):
                        indices = indices.flatten()
                    
                    best_idx = 0
                    best_conf = 0
                    for idx in indices:
                        if confidences[idx] > best_conf:
                            best_conf = confidences[idx]
                            best_idx = idx
                    
                    x1, y1, x2, y2 = boxes[best_idx]
                    confidence = confidences[best_idx]
                    
                    detections.append(DetectionResult(
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        confidence=float(confidence),
                        class_name="extension_box"
                    ))
                    
                    print(f"Best detection: conf={confidence:.3f}, bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        
        return detections
    
    except Exception as e:
        print(f"Error in postprocessing: {e}")
        import traceback
        traceback.print_exc()
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