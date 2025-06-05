import time
import torch
import torch.nn as nn
import onnxruntime
import numpy as np
import cv2
from collections import deque
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import copy
import traceback

from yolov8_utils import *

def frame_process(frame, input_shape=(640, 640)):
    """Process frame for YOLOv8 inference"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_shape)
    img = torch.from_numpy(img).float() / 255.0      
    img = img.permute(2, 0, 1)                        
    return img

class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = x.view(1, c1, 1, 1)
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape 
        return self.conv(
            x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)
        ).view(b, 4, a)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Convert distance predictions to bounding boxes"""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim) 
    return torch.cat((x1y1, x2y2), dim)  

def post_process(x):
    """Post-process YOLOv8 outputs"""
    dfl = DFL(16)
    anchors = torch.tensor(np.load("./anchors.npy", allow_pickle=True))
    strides = torch.tensor(np.load("./strides.npy", allow_pickle=True))

    box, cls = torch.cat(
        [xi.view(x[0].shape[0], 144, -1) for xi in x], dim=2
    ).split((16*4, 80), dim=1)

    dbox = dist2bbox(dfl(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
    y = torch.cat((dbox, cls.sigmoid()), dim=1)  
    return y, x

class OptimizedInferenceEngine:
    """Single session with optimized threading for NPU utilization"""
    
    def __init__(self, model_path, config_path):
        self.model_path = model_path
        self.config_path = config_path
        self.session = None
        self.session_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize session
        self._create_session()
        
    def _create_session(self):
        """Create optimized ONNX Runtime session"""
        try:
            with open(self.config_path, "r") as f:
                cfg = f.read()

            npu_opts = onnxruntime.SessionOptions()
            npu_opts.enable_profiling = False
            npu_opts.log_severity_level = 3
            npu_opts.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
            
            # Optimize provider options for better NPU utilization
            provider_opts = [{
                "config": cfg,
                "ai_analyzer_visualization": False,  # Disable for performance
                "ai_analyzer_profiling": False,
            }]

            self.session = onnxruntime.InferenceSession(
                self.model_path,
                providers=["VitisAIExecutionProvider"],
                sess_options=npu_opts,
                provider_options=provider_opts
            )
            
            print("Optimized inference session loaded successfully")
                
        except Exception as e:
            print(f"Error creating session: {e}")
            traceback.print_exc()
            raise
    
    def get_input_meta(self):
        """Get input metadata"""
        return self.session.get_inputs()[0]
    
    def infer_async(self, input_data, input_name):
        """Submit inference task asynchronously"""
        future = self.executor.submit(self._infer_sync, input_data, input_name)
        return future
    
    def _infer_sync(self, input_data, input_name):
        """Perform synchronous inference with thread safety"""
        try:
            with self.session_lock:
                outputs = self.session.run(None, {input_name: input_data})
            return outputs
            
        except Exception as e:
            print(f"Inference session error: {e}")
            traceback.print_exc()
            raise e
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)

class FrameBuffer:
    """Thread-safe frame buffer for pipeline processing"""
    
    def __init__(self, maxsize=3):  # Reduced buffer size
        self.queue = queue.Queue(maxsize=maxsize)
        self.latest_frame = None
        self.lock = threading.Lock()
    
    def put_frame(self, frame, frame_id):
        """Add frame to buffer (non-blocking)"""
        try:
            self.queue.put((frame.copy(), frame_id), block=False)
        except queue.Full:
            # Remove oldest frame and add new one
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put((frame.copy(), frame_id), block=False)
            except queue.Full:
                pass  # Skip if still full
        
        with self.lock:
            self.latest_frame = frame.copy()
    
    def get_frame(self, timeout=0.05):
        """Get frame from buffer"""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None, None
    
    def get_latest_frame(self):
        """Get the most recent frame for display"""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

def draw_fps_info(frame, fps, avg_fps, max_fps, min_fps, inference_fps=None):
    """Draw clean FPS information on frame"""
    overlay = frame.copy()
    
    box_height = 100 if inference_fps else 80
    box_width = 220
    
    cv2.rectangle(overlay, (10, 10), (10 + box_width, 10 + box_height), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + box_width, 10 + box_height), (100, 100, 100), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    fps_color = (0, 255, 0) if fps > 15 else (0, 255, 255) if fps > 10 else (0, 0, 255)
    text_color = (255, 255, 255)
    
    y_offset = 30
    cv2.putText(frame, f"Display FPS: {fps:.1f}", (20, y_offset), font, font_scale, fps_color, thickness)
    y_offset += 15
    cv2.putText(frame, f"Avg: {avg_fps:.1f}", (20, y_offset), font, font_scale, text_color, thickness)
    y_offset += 15
    cv2.putText(frame, f"Max: {max_fps:.1f}", (20, y_offset), font, font_scale, text_color, thickness)
    y_offset += 15
    cv2.putText(frame, f"Min: {min_fps:.1f}", (20, y_offset), font, font_scale, text_color, thickness)
    
    if inference_fps:
        y_offset += 15
        inf_color = (0, 255, 0) if inference_fps > 20 else (0, 255, 255) if inference_fps > 15 else (0, 0, 255)
        cv2.putText(frame, f"Inference: {inference_fps:.1f}", (20, y_offset), font, font_scale, inf_color, thickness)

def draw_detection_info(frame, num_detections, npu_utilization=None):
    """Draw detection count and NPU utilization information"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Detection count
    if num_detections > 0:
        text = f"Detections: {num_detections}"
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = frame.shape[1] - text_width - 20
        y = 30
        
        cv2.rectangle(frame, (x - 10, y - text_height - 5), (x + text_width + 10, y + 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (x - 10, y - text_height - 5), (x + text_width + 10, y + 5), (100, 100, 100), 2)
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), thickness)

def setup_camera():
    """Setup camera with higher FPS settings"""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    cap = None
    
    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            print(f"Camera opened with backend: {backend}")
            break
    
    if not cap or not cap.isOpened():
        raise RuntimeError("Cannot open webcam with any backend.")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60) 
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")
    
    return cap

class AsyncFrameCapture:
    """Asynchronous frame capture to decouple camera from display"""
    
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        
    def start(self):
        """Start capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()
        
    def _capture_frames(self):
        """Continuously capture frames"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.001)  # Brief pause on failure
                
    def get_frame(self):
        """Get latest frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
            
    def stop(self):
        """Stop capture thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

def inference_worker(inference_engine, frame_buffer, result_queue, inp_meta, stop_event):
    """Worker thread for running inference with better error handling"""
    inference_times = deque(maxlen=30)
    
    # Pre-load DFL and anchors once
    dfl = DFL(16)
    try:
        anchors = torch.tensor(np.load("./anchors.npy", allow_pickle=True))
        strides = torch.tensor(np.load("./strides.npy", allow_pickle=True))
    except Exception as e:
        print(f"Error loading anchors/strides: {e}")
        return
    
    print("Inference worker started")
    
    while not stop_event.is_set():
        try:
            frame_data = frame_buffer.get_frame(timeout=0.1)
            if frame_data[0] is None:
                continue
                
            frame, frame_id = frame_data
            
            if frame is None or frame.size == 0:
                continue
            
            start_time = time.time()
            
            # Preprocess frame
            try:
                im = frame_process(frame, input_shape=(640, 640))
                im = im.unsqueeze(0)
                nhwc = im.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
                nhwc = np.ascontiguousarray(nhwc)
            except Exception as e:
                print(f"Preprocessing error: {e}")
                continue
            
            # Run inference
            try:
                future = inference_engine.infer_async(nhwc, inp_meta.name)
                outputs = future.result(timeout=1.0)  # Increased timeout
            except Exception as e:
                print(f"Inference execution error: {e}")
                continue
            
            inference_time = time.time() - start_time
            if inference_time > 0:
                inference_times.append(1.0 / inference_time)
            
            # Convert outputs to PyTorch tensors
            try:
                outputs = [torch.tensor(o).permute(0, 3, 1, 2) for o in outputs]
            except Exception as e:
                print(f"Output conversion error: {e}")
                continue
            
            # Post-process with pre-loaded components
            try:
                box, cls = torch.cat(
                    [xi.view(outputs[0].shape[0], 144, -1) for xi in outputs], dim=2
                ).split((16*4, 80), dim=1)

                dbox = dist2bbox(dfl(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
                preds = torch.cat((dbox, cls.sigmoid()), dim=1)
                
                preds = non_max_suppression(
                    preds, 
                    conf_thres=0.25, 
                    iou_thres=0.7, 
                    agnostic=False, 
                    max_det=300, 
                    classes=None
                )
            except Exception as e:
                print(f"Post-processing error: {e}")
                continue
            
            # Calculate inference FPS
            avg_inference_fps = sum(inference_times) / len(inference_times) if inference_times else 0
            
            # Put result in queue
            try:
                result_queue.put((frame_id, preds[0], avg_inference_fps), block=False)
            except queue.Full:
                # Remove oldest result
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    result_queue.put((frame_id, preds[0], avg_inference_fps), block=False)
                except queue.Full:
                    pass  # Skip if still full
                    
        except Exception as e:
            print(f"Worker loop error: {e}")
            traceback.print_exc()
            time.sleep(0.1)  # Brief pause before retrying
    
    print("Inference worker stopped")

def main():
    """Main function with uncapped display FPS"""
    # Load COCO class names
    try:
        with open("coco.names", "r") as f:
            names = [n.strip() for n in f.readlines()]
    except FileNotFoundError:
        print("Warning: coco.names not found. Using default class names.")
        names = [f"class_{i}" for i in range(80)]
    
    # Initialize optimized inference engine
    onnx_model_path = "yolov8m.onnx"
    config_path = "./vaip_config.json"
    
    try:
        inference_engine = OptimizedInferenceEngine(onnx_model_path, config_path)
        inp_meta = inference_engine.get_input_meta()
        print("Optimized inference engine initialized successfully")
    except Exception as e:
        print(f"Error initializing inference engine: {e}")
        traceback.print_exc()
        return
    
    # Setup camera and async capture
    try:
        cap = setup_camera()
        async_capture = AsyncFrameCapture(cap)
        async_capture.start()
        print("Async frame capture started")
    except Exception as e:
        print(f"Camera setup error: {e}")
        return
        
    frame_buffer = FrameBuffer(maxsize=3)
    result_queue = queue.Queue(maxsize=5)
    
    # Create window with optimized settings
    cv2.namedWindow("YOLOv8 High-FPS Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 High-FPS Detection", 1280, 720)
    
    # FPS tracking
    fps_queue = deque(maxlen=30)
    prev_time = time.time()
    max_fps = 0.0
    min_fps = float("inf")
    
    # Start inference worker threads
    stop_event = threading.Event()
    num_workers = 2
    workers = []
    
    for i in range(num_workers):
        worker = threading.Thread(
            target=inference_worker,
            args=(inference_engine, frame_buffer, result_queue, inp_meta, stop_event),
            daemon=True
        )
        worker.start()
        workers.append(worker)
        print(f"Started inference worker {i+1}")
    
    # Color palette for different classes
    np.random.seed(42)
    colors = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(0, 255, size=(80, 3))]
    
    print("Starting high-FPS detection. Press 'q' to quit, 'r' to reset FPS stats.")
    
    frame_id = 0
    latest_detections = None
    latest_inference_fps = 0
    npu_utilization = 70.0
    
    # Display loop - now independent of camera capture rate
    try:
        while True:
            # Get latest frame from async capture
            current_frame = async_capture.get_frame()
            if current_frame is None:
                time.sleep(0.001)  # Brief pause if no frame
                continue

            # Calculate display FPS (now uncapped from camera)
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            
            fps_queue.append(fps)
            avg_fps = sum(fps_queue) / len(fps_queue)
            max_fps = max(fps, max_fps)
            min_fps = min(fps, min_fps)
            
            # Add frame to processing buffer (throttled)
            if frame_id % 2 == 0:  # Process every other frame to reduce load
                frame_buffer.put_frame(current_frame, frame_id)
            frame_id += 1
            
            # Check for new inference results
            try:
                while True:
                    result_frame_id, detections, inference_fps = result_queue.get_nowait()
                    latest_detections = detections
                    latest_inference_fps = inference_fps
                    npu_utilization = min(95.0, 60.0 + (inference_fps / 25.0) * 35.0)
            except queue.Empty:
                pass
            
            # Create annotated frame
            annotated_frame = current_frame.copy()
            h0, w0 = current_frame.shape[:2]
            scale_x, scale_y = w0 / 640, h0 / 640
            
            num_detections = 0

            # Draw latest detections (same code as before)
            if latest_detections is not None and latest_detections.numel() > 0:
                det = latest_detections.cpu().numpy()
                boxes_np = det[:, 0:4]
                scores_np = det[:, 4].astype(np.float32)
                class_ids_np = det[:, 5].astype(np.int32)
                
                num_detections = len(boxes_np)

                for i in range(len(boxes_np)):
                    x1, y1, x2, y2 = boxes_np[i]
                    cls_id = int(class_ids_np[i])
                    conf = float(scores_np[i])

                    x1n = int(x1 * scale_x)
                    y1n = int(y1 * scale_y)
                    x2n = int(x2 * scale_x)
                    y2n = int(y2 * scale_y)

                    color = colors[cls_id % len(colors)]
                    
                    cv2.rectangle(annotated_frame, (x1n, y1n), (x2n, y2n), color, 2)

                    label = f"{names[cls_id]} {conf:.2f}"
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    cv2.rectangle(
                        annotated_frame,
                        (x1n, y1n - label_height - 10),
                        (x1n + label_width, y1n),
                        color,
                        -1
                    )
                    
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1n, y1n - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

            # Draw UI elements
            draw_fps_info(annotated_frame, fps, avg_fps, max_fps, min_fps, latest_inference_fps)
            draw_detection_info(annotated_frame, num_detections, npu_utilization)

            # Display frame
            cv2.imshow("YOLOv8 High-FPS Detection", annotated_frame)

            # Handle key presses with minimal delay
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                fps_queue.clear()
                max_fps = 0.0
                min_fps = float("inf")
                print("FPS statistics reset")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        print("Shutting down...")
        stop_event.set()
        async_capture.stop()
        
        for worker in workers:
            worker.join(timeout=3.0)
        
        inference_engine.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")

if __name__ == "__main__":
    main()