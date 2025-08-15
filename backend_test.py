#!/usr/bin/env python3
"""
Backend API Testing for Extension Box Detection Application
Tests all FastAPI endpoints and WebSocket functionality
"""

import requests
import json
import base64
import asyncio
import websockets
import sys
import time
from datetime import datetime
from io import BytesIO
from PIL import Image
import numpy as np

class ExtensionBoxAPITester:
    def __init__(self, base_url="https://smartbox-scanner.preview.emergentagent.com"):
        self.base_url = base_url
        self.ws_url = base_url.replace('https', 'wss').replace('http', 'ws')
        self.tests_run = 0
        self.tests_passed = 0
        self.session = requests.Session()
        self.session.timeout = 30

    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED {details}")
        else:
            print(f"❌ {name} - FAILED {details}")
        return success

    def create_test_image(self, width=640, height=640):
        """Create a test image for detection testing"""
        # Create a simple test image with some geometric shapes
        image = Image.new('RGB', (width, height), color='white')
        pixels = np.array(image)
        
        # Add some colored rectangles to simulate extension boxes
        pixels[100:200, 100:300] = [255, 0, 0]  # Red rectangle
        pixels[300:400, 200:400] = [0, 255, 0]  # Green rectangle
        
        test_image = Image.fromarray(pixels)
        
        # Convert to base64
        buffer = BytesIO()
        test_image.save(buffer, format='JPEG', quality=80)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{image_base64}", width, height

    def test_health_endpoint(self):
        """Test the health check endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            
            if response.status_code == 200:
                data = response.json()
                model_status = data.get('model_status', 'unknown')
                
                if model_status == 'loaded':
                    return self.log_test("Health Check", True, f"- Model loaded successfully")
                else:
                    return self.log_test("Health Check", False, f"- Model not loaded: {model_status}")
            else:
                return self.log_test("Health Check", False, f"- HTTP {response.status_code}")
                
        except Exception as e:
            return self.log_test("Health Check", False, f"- Exception: {str(e)}")

    def test_detect_endpoint(self):
        """Test the POST /api/detect endpoint"""
        try:
            image_base64, width, height = self.create_test_image()
            
            payload = {
                "image_base64": image_base64,
                "image_width": width,
                "image_height": height
            }
            
            response = self.session.post(
                f"{self.base_url}/api/detect",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response structure
                required_fields = ['detections', 'image_width', 'image_height', 'processing_time']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return self.log_test("Detect Endpoint", False, f"- Missing fields: {missing_fields}")
                
                processing_time = data.get('processing_time', 0)
                detection_count = len(data.get('detections', []))
                
                return self.log_test("Detect Endpoint", True, 
                    f"- {detection_count} detections, {processing_time:.3f}s processing time")
            else:
                return self.log_test("Detect Endpoint", False, f"- HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            return self.log_test("Detect Endpoint", False, f"- Exception: {str(e)}")

    def test_detect_endpoint_invalid_data(self):
        """Test the detect endpoint with invalid data"""
        try:
            # Test with invalid base64
            payload = {
                "image_base64": "invalid_base64_data",
                "image_width": 640,
                "image_height": 640
            }
            
            response = self.session.post(
                f"{self.base_url}/api/detect",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            # Should return error status
            if response.status_code >= 400:
                return self.log_test("Detect Endpoint (Invalid Data)", True, f"- Correctly rejected invalid data with HTTP {response.status_code}")
            else:
                return self.log_test("Detect Endpoint (Invalid Data)", False, f"- Should have rejected invalid data but got HTTP {response.status_code}")
                
        except Exception as e:
            return self.log_test("Detect Endpoint (Invalid Data)", False, f"- Exception: {str(e)}")

    async def test_websocket_connection(self):
        """Test WebSocket connection and basic functionality"""
        try:
            ws_uri = f"{self.ws_url}/api/ws/detect"
            print(f"🔍 Testing WebSocket connection to: {ws_uri}")
            
            async with websockets.connect(ws_uri, timeout=10) as websocket:
                # Test connection
                self.log_test("WebSocket Connection", True, "- Connected successfully")
                
                # Send test image
                image_base64, width, height = self.create_test_image()
                
                message = {
                    "type": "image",
                    "image": image_base64,
                    "width": width,
                    "height": height
                }
                
                await websocket.send(json.dumps(message))
                self.log_test("WebSocket Send", True, "- Sent test image")
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                    data = json.loads(response)
                    
                    if data.get("type") == "detection_result":
                        detections = data.get("detections", [])
                        self.log_test("WebSocket Response", True, 
                            f"- Received {len(detections)} detections")
                        return True
                    elif data.get("type") == "error":
                        self.log_test("WebSocket Response", False, 
                            f"- Error response: {data.get('message', 'Unknown error')}")
                        return False
                    else:
                        self.log_test("WebSocket Response", False, 
                            f"- Unexpected response type: {data.get('type', 'unknown')}")
                        return False
                        
                except asyncio.TimeoutError:
                    self.log_test("WebSocket Response", False, "- Timeout waiting for response")
                    return False
                    
        except Exception as e:
            self.log_test("WebSocket Connection", False, f"- Exception: {str(e)}")
            return False

    def test_cors_headers(self):
        """Test CORS headers are properly set"""
        try:
            response = self.session.options(f"{self.base_url}/api/health")
            
            cors_headers = {
                'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
                'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
                'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
            }
            
            if cors_headers['Access-Control-Allow-Origin']:
                return self.log_test("CORS Headers", True, f"- Origin: {cors_headers['Access-Control-Allow-Origin']}")
            else:
                return self.log_test("CORS Headers", False, "- Missing CORS headers")
                
        except Exception as e:
            return self.log_test("CORS Headers", False, f"- Exception: {str(e)}")

    async def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting Extension Box Detection Backend Tests")
        print(f"📍 Testing backend at: {self.base_url}")
        print("=" * 60)
        
        # Basic API tests
        self.test_health_endpoint()
        self.test_detect_endpoint()
        self.test_detect_endpoint_invalid_data()
        self.test_cors_headers()
        
        # WebSocket tests
        await self.test_websocket_connection()
        
        # Print summary
        print("=" * 60)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All backend tests passed!")
            return True
        else:
            failed_tests = self.tests_run - self.tests_passed
            print(f"⚠️  {failed_tests} test(s) failed")
            return False

async def main():
    """Main test runner"""
    tester = ExtensionBoxAPITester()
    success = await tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)