import time
import torch
import torch.nn as nn
import onnxruntime
import numpy as np
import cv2
from collections import deque
import threading
import queue

from yolov8_utils import *

# --- NEW: Thread-Safe Frame Handling ---

class FrameReader(threading.Thread):
    """
    Reads frames from a camera stream and continuously updates a shared frame variable.
    This runs as fast as the camera can provide frames.
    """
    def __init__(self, cap):
        super().__init__()
        self.daemon = True
        self.cap = cap
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("FrameReader: Failed to grab frame, stopping.")
                self.running = False
                continue
            
            with self.lock:
                self.frame = frame
        self.cap.release()
        print("FrameReader thread stopped.")

    def get_frame(self):
        """Returns a thread-safe copy of the latest frame."""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False

class DetectionWorker(threading.Thread):
    """
    Runs YOLOv8 detection and tracks detailed performance statistics (min, avg, max FPS).
    """
    def __init__(self, in_queue, out_queue, model_session, input_meta):
        super().__init__()
        self.daemon = True
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.npu_session = model_session
        self.inp_meta = input_meta
        self.running = True
        
        # --- NEW: Expanded FPS Tracking ---
        self.fps_queue = deque(maxlen=30)
        self.min_fps = float('inf')
        self.max_fps = 0.0
        
        # Pre-loaded components for post-processing
        self.dfl = DFL(16)
        self.anchors = torch.tensor(np.load("./anchors.npy", allow_pickle=True))
        self.strides = torch.tensor(np.load("./strides.npy", allow_pickle=True))

    def run(self):
        while self.running:
            try:
                frame = self.in_queue.get(timeout=1)
            except queue.Empty:
                continue

            start_time = time.time()
            
            # --- Inference Pipeline ---
            im = frame_process(frame, input_shape=(640, 640))
            nhwc = im.unsqueeze(0).permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
            
            outputs = self.npu_session.run(None, {self.inp_meta.name: np.ascontiguousarray(nhwc)})
            outputs = [torch.tensor(o).permute(0, 3, 1, 2) for o in outputs]
            
            preds = self.post_process(outputs)
            preds = non_max_suppression(
                preds, conf_thres=0.25, iou_thres=0.7, agnostic=False, max_det=300
            )
            
            # --- NEW: Update all FPS stats ---
            end_time = time.time()
            current_fps = 1.0 / (end_time - start_time)
            self.fps_queue.append(current_fps)
            self.min_fps = min(self.min_fps, current_fps)
            self.max_fps = max(self.max_fps, current_fps)
            avg_fps = sum(self.fps_queue) / len(self.fps_queue)

            # --- NEW: Pass all stats in the output queue ---
            stats_tuple = (avg_fps, self.min_fps, self.max_fps)
            if not self.out_queue.full():
                self.out_queue.put((preds[0], stats_tuple))
            
            self.in_queue.task_done()

    def post_process(self, x):
        box, cls = torch.cat([xi.view(x[0].shape[0], 144, -1) for xi in x], dim=2).split((64, 80), dim=1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        return torch.cat((dbox, cls.sigmoid()), dim=1), x

    def stop(self):
        self.running = False


# --- Main Function (Corrected Architecture) ---

def main():
    """Main function with CORRECT and robust FPS capping."""
    
    # --- FPS Capping Configuration ---
    CAP_DISPLAY_FPS = True  # Set to True to cap FPS, False for unlimited
    TARGET_FPS = 60.0
    
    # --- Setup (Model, Camera, etc.) ---
    try:
        with open("coco.names", "r") as f:
            names = [n.strip() for n in f.readlines()]
        np.random.seed(42)
        colors = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(0, 255, size=(len(names), 3))]
    except Exception as e:
        print(f"Error loading assets: {e}"); return

    try:
        npu_opts = onnxruntime.SessionOptions()
        with open("./vaip_config.json", "r") as f: cfg = f.read()
        provider_opts = [{"config": cfg}]
        npu_session = onnxruntime.InferenceSession("yolov8m.onnx", providers=["VitisAIExecutionProvider"], sess_options=npu_opts, provider_options=provider_opts)
        inp_meta = npu_session.get_inputs()[0]
        print("Model loaded successfully with VitisAI provider")
    except Exception as e:
        print(f"Error loading model: {e}"); return

    cap = setup_camera()
    
    # --- Threading and Queues Setup ---
    detection_in_queue = queue.Queue(maxsize=1)
    detection_out_queue = queue.Queue(maxsize=1)

    frame_reader = FrameReader(cap)
    detection_worker = DetectionWorker(detection_in_queue, detection_out_queue, npu_session, inp_meta)
    
    frame_reader.start()
    detection_worker.start()
    time.sleep(1) 

    # --- Main Display Loop ---
    cv2.namedWindow("YOLOv8 Real-time Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 Real-time Detection", 1280, 720)

    display_fps_queue = deque(maxlen=60)
    prev_loop_end_time = time.time()
    
    latest_detections = None
    detection_stats = (0.0, float('inf'), 0.0) # Initialize min FPS properly

    print("Starting real-time detection. Press 'q' to quit.")
    
    try:
        while True:
            # --- START: Corrected FPS Capping and Measurement ---
            loop_start_time = time.time()
            target_frame_time = 1.0 / TARGET_FPS if CAP_DISPLAY_FPS else 0

            # 1. Grab the latest frame
            frame = frame_reader.get_frame()
            if frame is None:
                if not frame_reader.running: break
                time.sleep(0.001)
                continue

            # 2. Feed the detector
            if not detection_in_queue.full():
                detection_in_queue.put(frame)

            # 3. Check for new results
            try:
                latest_detections, detection_stats = detection_out_queue.get_nowait()
            except queue.Empty:
                pass

            # 4. Draw all visuals
            annotated_frame = frame
            num_detections = draw_detections(annotated_frame, latest_detections, names, colors) if latest_detections is not None else 0
            
            # We calculate the current average display FPS *before* drawing it
            avg_display_fps = sum(display_fps_queue) / len(display_fps_queue) if display_fps_queue else 0
            draw_fps_info(annotated_frame, avg_display_fps, detection_stats)
            draw_detection_info(annotated_frame, num_detections)

            # 5. Display the frame
            cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 6. Enforce FPS Cap
            if CAP_DISPLAY_FPS:
                # Calculate how long the loop took and how long to sleep
                elapsed_time = time.time() - loop_start_time
                sleep_time = target_frame_time - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # 7. Calculate final, accurate FPS for this iteration and add to queue
            # This measures the time from the end of the last loop to the end of this one
            current_loop_end_time = time.time()
            actual_frame_time = current_loop_end_time - prev_loop_end_time
            prev_loop_end_time = current_loop_end_time
            if actual_frame_time > 0:
                display_fps_queue.append(1.0 / actual_frame_time)

            # --- END: Corrected FPS Capping and Measurement ---

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        print("Stopping threads...")
        frame_reader.stop()
        detection_worker.stop()
        frame_reader.join()
        detection_worker.join()
        cv2.destroyAllWindows()
        print("Application finished.")

# --- Helper Functions (Refactored and Unchanged) ---

def draw_detections(frame, detections, names, colors):
    if detections is None or detections.numel() == 0:
        return 0
        
    h0, w0 = frame.shape[:2]
    scale_x, scale_y = w0 / 640, h0 / 640
    
    det_np = detections.cpu().numpy()
    boxes, scores, class_ids = det_np[:, 0:4], det_np[:, 4], det_np[:, 5].astype(int)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cls_id, conf = class_ids[i], scores[i]
        
        x1n, y1n = int(x1 * scale_x), int(y1 * scale_y)
        x2n, y2n = int(x2 * scale_x), int(y2 * scale_y)
        
        color = colors[cls_id % len(colors)]
        cv2.rectangle(frame, (x1n, y1n), (x2n, y2n), color, 2)
        
        label = f"{names[cls_id]} {conf:.2f}"
        (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1n, y1n - lh - 10), (x1n + lw, y1n), color, -1)
        cv2.putText(frame, label, (x1n, y1n - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return len(boxes)

def setup_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW is often faster on Windows
    if not cap.isOpened():
        print("DSHOW backend failed, trying default...")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
         raise RuntimeError("Cannot open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera reports: {actual_width}x{actual_height} @ {actual_fps} FPS")
    return cap

def frame_process(frame, input_shape=(640, 640)):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_shape)
    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1)
    return img

class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__(); self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float); self.conv.weight.data[:] = x.view(1, c1, 1, 1); self.c1 = c1
    def forward(self, x):
        b, c, a = x.shape; return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = torch.split(distance, 2, dim); x1y1 = anchor_points - lt; x2y2 = anchor_points + rb
    if xywh: c_xy = (x1y1 + x2y2) / 2; wh = x2y2 - x1y1; return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)

def draw_fps_info(frame, display_fps, det_stats):
    """Draws two separate, styled boxes for Display and Detection FPS."""
    det_avg_fps, det_min_fps, det_max_fps = det_stats
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # --- 1. Display FPS Box (Top) ---
    box1_pos = (10, 10)
    box1_size = (200, 50)
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, box1_pos, (box1_pos[0] + box1_size[0], box1_pos[1] + box1_size[1]), 
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Border
    cv2.rectangle(frame, box1_pos, (box1_pos[0] + box1_size[0], box1_pos[1] + box1_size[1]), 
                  (100, 100, 100), 2)

    # Text
    cv2.putText(frame, "Display", (box1_pos[0] + 10, box1_pos[1] + 20), font, font_scale, (255, 255, 255), thickness)
    display_fps_color = (0, 255, 0) if display_fps > 50 else (0, 255, 255) if display_fps > 30 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {display_fps:.1f}", (box1_pos[0] + 10, box1_pos[1] + 40), font, font_scale, display_fps_color, thickness)


    # --- 2. Detection FPS Box (Bottom) ---
    box2_pos = (10, box1_pos[1] + box1_size[1] + 10) # Position below the first box
    box2_size = (200, 95)
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, box2_pos, (box2_pos[0] + box2_size[0], box2_pos[1] + box2_size[1]), 
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Border
    cv2.rectangle(frame, box2_pos, (box2_pos[0] + box2_size[0], box2_pos[1] + box2_size[1]), 
                  (100, 100, 100), 2)
    
    # Text
    cv2.putText(frame, "Detection", (box2_pos[0] + 10, box2_pos[1] + 20), font, font_scale, (255, 255, 255), thickness)
    
    # Use the average detection FPS for the main "FPS" line to match the style
    det_fps_color = (0, 255, 0) if det_avg_fps > 15 else (0, 255, 255) if det_avg_fps > 10 else (0, 0, 255)
    text_color = (255, 255, 255)
    
    y_offset = box2_pos[1] + 40
    cv2.putText(frame, f"FPS: {det_avg_fps:.1f}", (box2_pos[0] + 10, y_offset), font, font_scale, det_fps_color, thickness)
    cv2.putText(frame, f"Avg: {det_avg_fps:.1f}", (box2_pos[0] + 10, y_offset + 15), font, font_scale, text_color, thickness)
    cv2.putText(frame, f"Max: {det_max_fps:.1f}", (box2_pos[0] + 10, y_offset + 30), font, font_scale, text_color, thickness)
    cv2.putText(frame, f"Min: {det_min_fps:.1f}", (box2_pos[0] + 10, y_offset + 45), font, font_scale, text_color, thickness)

def draw_detection_info(frame, num_detections):
    if num_detections <= 0: return
    text = f"Detections: {num_detections}"; font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = frame.shape[1] - tw - 20, 35
    cv2.rectangle(frame, (x - 10, y - th - 5), (x + tw + 10, y + 5), (0, 0, 0), -1)
    cv2.rectangle(frame, (x - 10, y - th - 5), (x + tw + 10, y + 5), (100, 100, 100), 2)
    cv2.putText(frame, text, (x, y), font, scale, (0, 255, 0), thick)

if __name__ == "__main__":
    main()