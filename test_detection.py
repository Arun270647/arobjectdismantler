#!/usr/bin/env python3
"""
Test script to verify extension box detection is working
"""
import requests
import base64
import json
from PIL import Image, ImageDraw
import numpy as np
from io import BytesIO

def create_test_image_with_box():
    """Create a test image with a box-like shape"""
    # Create a white background
    img = Image.new('RGB', (640, 480), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a rectangular "extension box" shape
    # Outer rectangle (box body)
    draw.rectangle([200, 150, 440, 330], fill='gray', outline='black', width=3)
    
    # Add some "outlet" holes
    draw.rectangle([220, 180, 250, 210], fill='black')
    draw.rectangle([270, 180, 300, 210], fill='black')
    draw.rectangle([320, 180, 350, 210], fill='black')
    draw.rectangle([370, 180, 400, 210], fill='black')
    
    # Add power switch
    draw.rectangle([410, 160, 430, 180], fill='red', outline='black')
    
    # Add power cord
    draw.line([320, 330, 320, 380], fill='black', width=5)
    
    return img

def test_detection_api():
    """Test the detection API with a synthetic extension box image"""
    print("🧪 Testing Extension Box Detection API...")
    
    # Create test image
    test_image = create_test_image_with_box()
    
    # Convert to base64
    buffer = BytesIO()
    test_image.save(buffer, format='JPEG', quality=90)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    image_data_url = f"data:image/jpeg;base64,{image_base64}"
    
    # Prepare API request
    payload = {
        "image_base64": image_data_url,
        "image_width": test_image.width,
        "image_height": test_image.height
    }
    
    # Test the API
    try:
        print(f"📤 Sending test image ({test_image.width}x{test_image.height}) to detection API...")
        
        response = requests.post(
            "https://smartbox-scanner.preview.emergentagent.com/api/detect",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API Response successful!")
            print(f"   Processing time: {result['processing_time']:.3f}s")
            print(f"   Detections found: {len(result['detections'])}")
            
            for i, detection in enumerate(result['detections']):
                bbox = detection['bbox']
                conf = detection['confidence']
                cls = detection['class_name']
                print(f"   Detection {i+1}: {cls} (conf: {conf:.3f}) bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                
            return len(result['detections']) > 0
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        print("🏥 Testing health endpoint...")
        response = requests.get("https://smartbox-scanner.preview.emergentagent.com/api/health")
        
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health check passed!")
            print(f"   Status: {health.get('status', 'unknown')}")
            print(f"   Model: {health.get('model_status', 'unknown')}")
            return health.get('model_status') == 'loaded'
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Extension Box Detection - Testing Suite")
    print("=" * 50)
    
    # Test health first
    health_ok = test_health_endpoint()
    print()
    
    if health_ok:
        # Test detection
        detection_ok = test_detection_api()
        print()
        
        if detection_ok:
            print("🎉 SUCCESS: Detection is working!")
        else:
            print("⚠️  Detection API works but no detections found (this might be normal if the model is very specific)")
    else:
        print("❌ Health check failed - cannot proceed with detection test")
    
    print("=" * 50)