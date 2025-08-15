import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, CameraOff, RotateCcw, Zap, AlertCircle, CheckCircle2, Smartphone, Monitor } from 'lucide-react';
import { Button } from './components/ui/button';
import { Card } from './components/ui/card';
import { Badge } from './components/ui/badge';
import './App.css';

const App = () => {
  // State management
  const [isStreaming, setIsStreaming] = useState(false);
  const [detections, setDetections] = useState([]);
  const [currentCamera, setCurrentCamera] = useState('environment'); // 'user' for front, 'environment' for back
  const [error, setError] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [detectionStatus, setDetectionStatus] = useState('idle'); // 'idle', 'detecting', 'found'
  const [isMobile, setIsMobile] = useState(false);
  const [streamStats, setStreamStats] = useState({ fps: 0, detectionCount: 0 });

  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const wsRef = useRef(null);
  const detectionIntervalRef = useRef(null);

  // Backend URL from environment
  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
  const wsUrl = backendUrl.replace('http', 'ws').replace('https', 'wss');

  // Detect mobile device
  useEffect(() => {
    const checkMobile = () => {
      const mobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      setIsMobile(mobile);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Initialize WebSocket connection
  const initWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      wsRef.current = new WebSocket(`${wsUrl}/api/ws/detect`);
      
      wsRef.current.onopen = () => {
        setIsConnected(true);
        setError('');
        console.log('WebSocket connected');
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'detection_result') {
            setDetections(data.detections);
            setDetectionStatus(data.detections.length > 0 ? 'found' : 'detecting');
            
            // Update stats
            setStreamStats(prev => ({
              ...prev,
              detectionCount: prev.detectionCount + data.detections.length
            }));
          } else if (data.type === 'error') {
            console.error('Detection error:', data.message);
            setDetectionStatus('idle');
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      wsRef.current.onclose = () => {
        setIsConnected(false);
        console.log('WebSocket disconnected');
        // Attempt to reconnect after 3 seconds
        setTimeout(() => {
          if (isStreaming) initWebSocket();
        }, 3000);
      };

      wsRef.current.onerror = (err) => {
        console.error('WebSocket error:', err);
        setError('Connection error. Retrying...');
        setIsConnected(false);
      };

    } catch (err) {
      console.error('Error initializing WebSocket:', err);
      setError('Failed to initialize connection');
    }
  }, [wsUrl, isStreaming]);

  // Start camera stream
  const startStream = async () => {
    try {
      setError('');
      
      // Stop existing stream if any
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      // Request camera access
      const constraints = {
        video: {
          facingMode: currentCamera,
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsStreaming(true);
      setDetectionStatus('detecting');
      
      // Initialize WebSocket
      initWebSocket();
      
      // Start detection loop (every 500ms as requested)
      startDetectionLoop();

    } catch (err) {
      console.error('Error accessing camera:', err);
      setError(`Camera access failed: ${err.message}`);
      setIsStreaming(false);
    }
  };

  // Stop camera stream
  const stopStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
    }

    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
    }

    setIsStreaming(false);
    setIsConnected(false);
    setDetections([]);
    setDetectionStatus('idle');
    setError('');
  };

  // Toggle camera (front/back for mobile)
  const toggleCamera = async () => {
    const newCamera = currentCamera === 'user' ? 'environment' : 'user';
    setCurrentCamera(newCamera);
    
    if (isStreaming) {
      // Restart stream with new camera
      stopStream();
      setTimeout(() => {
        setCurrentCamera(newCamera);
        startStream();
      }, 100);
    }
  };

  // Capture frame and send for detection
  const captureAndDetect = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !isConnected) return;

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      // Set canvas size to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw current frame
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert to base64
      const imageData = canvas.toDataURL('image/jpeg', 0.8);

      // Send to WebSocket
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        const message = {
          type: 'image',
          image: imageData,
          width: canvas.width,
          height: canvas.height
        };
        wsRef.current.send(JSON.stringify(message));
      }

    } catch (err) {
      console.error('Error capturing frame:', err);
    }
  }, [isConnected]);

  // Start detection loop
  const startDetectionLoop = () => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
    }

    detectionIntervalRef.current = setInterval(() => {
      captureAndDetect();
    }, 500); // Every 0.5 seconds as requested
  };

  // Draw detection overlays
  const drawDetections = () => {
    if (!videoRef.current || !canvasRef.current || detections.length === 0) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Clear previous overlays
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw each detection
    detections.forEach((detection, index) => {
      const [x1, y1, x2, y2] = detection.bbox;
      const confidence = detection.confidence;

      // Scale coordinates to canvas size
      const scaleX = canvas.width / video.videoWidth;
      const scaleY = canvas.height / video.videoHeight;

      const scaledX1 = x1 * scaleX;
      const scaledY1 = y1 * scaleY;
      const scaledX2 = x2 * scaleX;
      const scaledY2 = y2 * scaleY;

      // Draw bounding box
      ctx.strokeStyle = '#00ff41';
      ctx.lineWidth = 3;
      ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);

      // Draw confidence label
      const label = `Extension Box ${(confidence * 100).toFixed(1)}%`;
      ctx.fillStyle = '#00ff41';
      ctx.font = '16px Inter, sans-serif';
      ctx.fillRect(scaledX1, scaledY1 - 25, ctx.measureText(label).width + 10, 25);
      
      ctx.fillStyle = '#000';
      ctx.fillText(label, scaledX1 + 5, scaledY1 - 8);
    });
  };

  // Update overlay when detections change
  useEffect(() => {
    drawDetections();
  }, [detections]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStream();
    };
  }, []);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">
              <Zap className="w-6 h-6" />
            </div>
            <h1 className="app-title">SmartBox Scanner</h1>
          </div>
          
          <div className="device-indicator">
            {isMobile ? (
              <div className="device-badge mobile">
                <Smartphone className="w-4 h-4" />
                <span>Mobile</span>
              </div>
            ) : (
              <div className="device-badge desktop">
                <Monitor className="w-4 h-4" />
                <span>Desktop</span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Status Panel */}
        <Card className="status-panel">
          <div className="status-content">
            <div className="connection-status">
              <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
                {isConnected ? (
                  <CheckCircle2 className="w-5 h-5" />
                ) : (
                  <AlertCircle className="w-5 h-5" />
                )}
              </div>
              <span className="status-text">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>

            <div className="detection-status">
              <Badge 
                variant={detectionStatus === 'found' ? 'default' : 'secondary'}
                className={`detection-badge ${detectionStatus}`}
              >
                {detectionStatus === 'idle' && 'Ready'}
                {detectionStatus === 'detecting' && 'Scanning...'}
                {detectionStatus === 'found' && `Found ${detections.length} Box${detections.length !== 1 ? 'es' : ''}`}
              </Badge>
            </div>

            <div className="stats">
              <span className="stat">
                Detections: {streamStats.detectionCount}
              </span>
            </div>
          </div>
        </Card>

        {/* Camera Container */}
        <div className="camera-container">
          <div className="video-wrapper">
            <video
              ref={videoRef}
              className="video-stream"
              autoPlay
              playsInline
              muted
            />
            <canvas
              ref={canvasRef}
              className="detection-overlay"
            />
            
            {!isStreaming && (
              <div className="camera-placeholder">
                <Camera className="w-16 h-16 opacity-50" />
                <p className="placeholder-text">
                  Click "Start Detection" to begin scanning for extension boxes
                </p>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="camera-controls">
            <Button
              onClick={isStreaming ? stopStream : startStream}
              size="lg"
              className={`primary-button ${isStreaming ? 'stop' : 'start'}`}
            >
              {isStreaming ? (
                <>
                  <CameraOff className="w-5 h-5" />
                  Stop Detection
                </>
              ) : (
                <>
                  <Camera className="w-5 h-5" />
                  Start Detection
                </>
              )}
            </Button>

            {isMobile && (
              <Button
                onClick={toggleCamera}
                variant="outline"
                size="lg"
                disabled={!isStreaming}
                className="toggle-camera-button"
              >
                <RotateCcw className="w-5 h-5" />
                {currentCamera === 'environment' ? 'Switch to Front' : 'Switch to Back'}
              </Button>
            )}
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <Card className="error-card">
            <div className="error-content">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <p className="error-message">{error}</p>
            </div>
          </Card>
        )}

        {/* Detection Results */}
        {detections.length > 0 && (
          <Card className="results-panel">
            <h3 className="results-title">Detection Results</h3>
            <div className="results-list">
              {detections.map((detection, index) => (
                <div key={index} className="detection-item">
                  <div className="detection-info">
                    <span className="detection-class">{detection.class_name}</span>
                    <span className="detection-confidence">
                      {(detection.confidence * 100).toFixed(1)}% confidence
                    </span>
                  </div>
                  <div className="bbox-info">
                    Box: [{detection.bbox.map(coord => coord.toFixed(0)).join(', ')}]
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p className="footer-text">
          AI-powered extension box detection • Real-time scanning every 0.5 seconds
        </p>
      </footer>
    </div>
  );
};

export default App;