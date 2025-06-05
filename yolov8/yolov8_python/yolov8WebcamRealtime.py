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
    Runs YOLOv8 detection on frames from an input queue and places results in an output queue.
    """
    def __init__(self, in_queue, out_queue, model_session, input_meta):
        super().__init__()
        self.daemon = True
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.npu_session = model_session
        self.inp_meta = input_meta
        self.running = True
        self.fps_queue = deque(maxlen=30)
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
            
            self.fps_queue.append(1.0 / (time.time() - start_time))

            # Put results in the output queue
            if not self.out_queue.full():
                self.out_queue.put((preds[0], self.get_average_fps()))
            
            self.in_queue.task_done()

    def post_process(self, x):
        box, cls = torch.cat([xi.view(x[0].shape[0], 144, -1) for xi in x], dim=2).split((64, 80), dim=1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        return torch.cat((dbox, cls.sigmoid()), dim=1), x

    def get_average_fps(self):
        return sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0.0

    def stop(self):
        self.running = False


# --- Main Function (Corrected Architecture) ---

def main():
    """Main function with truly decoupled display and detection."""
    # --- Setup (Model, Camera, etc.) ---
    try:
        with open("coco.names", "r") as f:
            names = [n.strip() for n in f.readlines()]
        np.random.seed(42)
        colors = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(0, 255, size=(len(names), 3))]
    except Exception as e:
        print(f"Error loading assets: {e}")
        return

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

    # Wait briefly for the first frame to be read
    time.sleep(1) 

    # --- Main Display Loop ---
    cv2.namedWindow("YOLOv8 Real-time Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 Real-time Detection", 1280, 720)

    display_fps_queue = deque(maxlen=60)
    prev_time = time.time()
    
    latest_detections = None
    detection_fps = 0.0

    print("Starting real-time detection. Press 'q' to quit.")
    
    try:
        while True:
            # 1. Grab the latest frame from the reader thread
            frame = frame_reader.get_frame()
            if frame is None:
                if not frame_reader.running: break
                time.sleep(0.01)
                continue

            # 2. Feed the detector (non-blocking)
            if not detection_in_queue.full():
                detection_in_queue.put(frame)

            # 3. Check for new detection results (non-blocking)
            try:
                latest_detections, detection_fps = detection_out_queue.get_nowait()
            except queue.Empty:
                pass # No new results, continue using the old ones

            # 4. Draw detections on the frame
            annotated_frame = frame # No copy needed since get_frame() already provided one
            num_detections = 0
            if latest_detections is not None:
                num_detections = draw_detections(annotated_frame, latest_detections, names, colors)
            
            # 5. Draw UI Info
            curr_time = time.time()
            delta_time = curr_time - prev_time
            prev_time = curr_time

            if delta_time > 0:
                display_fps = 1.0 / delta_time
                display_fps_queue.append(display_fps)

            avg_display_fps = 0.0
            if display_fps_queue:
                avg_display_fps = sum(display_fps_queue) / len(display_fps_queue)
            
            draw_fps_info(annotated_frame, avg_display_fps, detection_fps)
            draw_detection_info(annotated_frame, num_detections)

            # 6. Display the frame
            cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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

def draw_fps_info(frame, display_fps, detection_fps):
    overlay = frame.copy(); box_height, box_width = 60, 220
    cv2.rectangle(overlay, (10, 10), (10 + box_width, 10 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + box_width, 10 + box_height), (100, 100, 100), 2)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    cv2.putText(frame, f"Display FPS: {display_fps:.1f}", (20, 35), font, scale, (0, 255, 0), thick)
    cv2.putText(frame, f"Detection FPS: {detection_fps:.1f}", (20, 60), font, scale, (0, 255, 255), thick)

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