import time
import torch
import torch.nn as nn
import onnxruntime
import numpy as np
import cv2
from collections import deque

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

def draw_fps_info(frame, fps, avg_fps, max_fps, min_fps):
    """Draw clean FPS information on frame"""
    # Create semi-transparent overlay for FPS info
    overlay = frame.copy()
    
    # FPS info box dimensions
    box_height = 80
    box_width = 200
    
    # Draw rounded rectangle background
    cv2.rectangle(overlay, (10, 10), (10 + box_width, 10 + box_height), (0, 0, 0), -1)
    
    # Blend with original frame for transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Add border
    cv2.rectangle(frame, (10, 10), (10 + box_width, 10 + box_height), (100, 100, 100), 2)
    
    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # FPS text colors
    fps_color = (0, 255, 0) if fps > 15 else (0, 255, 255) if fps > 10 else (0, 0, 255)
    text_color = (255, 255, 255)
    
    # Draw FPS information
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), font, font_scale, fps_color, thickness)
    cv2.putText(frame, f"Avg: {avg_fps:.1f}", (20, 45), font, font_scale, text_color, thickness)
    cv2.putText(frame, f"Max: {max_fps:.1f}", (20, 60), font, font_scale, text_color, thickness)
    cv2.putText(frame, f"Min: {min_fps:.1f}", (20, 75), font, font_scale, text_color, thickness)

def draw_detection_info(frame, num_detections):
    """Draw detection count information"""
    if num_detections > 0:
        # Detection info box
        text = f"Detections: {num_detections}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position at top right
        x = frame.shape[1] - text_width - 20
        y = 30
        
        # Draw background
        cv2.rectangle(frame, (x - 10, y - text_height - 5), (x + text_width + 10, y + 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (x - 10, y - text_height - 5), (x + text_width + 10, y + 5), (100, 100, 100), 2)
        
        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), thickness)

def setup_camera():
    """Setup camera with optimal settings"""
    # Try different backends for better performance
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    cap = None
    
    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            print(f"Camera opened with backend: {backend}")
            break
    
    if not cap or not cap.isOpened():
        raise RuntimeError("Cannot open webcam with any backend.")
    
    # Set camera properties for 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # Verify actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")
    
    return cap

def main():
    """Main function for YOLOv8 real-time detection"""
    # Load COCO class names
    try:
        with open("coco.names", "r") as f:
            names = [n.strip() for n in f.readlines()]
    except FileNotFoundError:
        print("Warning: coco.names not found. Using default class names.")
        names = [f"class_{i}" for i in range(80)]
    
    # Setup ONNX Runtime session
    onnx_model_path = "yolov8m.onnx"
    try:
        npu_opts = onnxruntime.SessionOptions()
        with open("./vaip_config.json", "r") as f:
            cfg = f.read()

        provider_opts = [{
            "config": cfg,
            "ai_analyzer_visualization": True,
            "ai_analyzer_profiling": True,
        }]

        npu_session = onnxruntime.InferenceSession(
            onnx_model_path,
            providers=["VitisAIExecutionProvider"],
            sess_options=npu_opts,
            provider_options=provider_opts
        )
        print("Model loaded successfully with VitisAI provider")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    inp_meta = npu_session.get_inputs()[0]
    
    # Setup camera
    cap = setup_camera()
    
    # Create window with specific size
    cv2.namedWindow("YOLOv8 Real-time Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 Real-time Detection", 1280, 720)
    
    # FPS tracking
    fps_queue = deque(maxlen=30)  # Rolling average over 30 frames
    prev_time = time.time()
    max_fps = 0.0
    min_fps = float("inf")
    
    # Color palette for different classes
    np.random.seed(42)  # For consistent colors
    colors = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(0, 255, size=(80, 3))]
    
    print("Starting real-time detection. Press 'q' to quit, 'r' to reset FPS stats.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            
            fps_queue.append(fps)
            avg_fps = sum(fps_queue) / len(fps_queue)
            max_fps = max(fps, max_fps)
            min_fps = min(fps, min_fps)

            # Preprocess frame
            im = frame_process(frame, input_shape=(640, 640))
            im = im.unsqueeze(0)

            # Convert to NHWC format for ONNX
            nhwc = im.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
            nhwc = np.ascontiguousarray(nhwc)

            # Run inference
            try:
                outputs = npu_session.run(None, {inp_meta.name: nhwc})
            except Exception as e:
                print(f"Inference error: {e}")
                continue

            # Convert outputs to PyTorch tensors
            outputs = [torch.tensor(o).permute(0, 3, 1, 2) for o in outputs]

            # Post-process
            preds = post_process(outputs)
            preds = non_max_suppression(
                preds, 
                conf_thres=0.25, 
                iou_thres=0.7, 
                agnostic=False, 
                max_det=300, 
                classes=None
            )

            det = preds[0]
            
            # Create annotated frame
            annotated_frame = frame.copy()
            h0, w0 = frame.shape[:2]
            scale_x, scale_y = w0 / 640, h0 / 640
            
            num_detections = 0

            # Draw detections
            if det is not None and det.numel() > 0:
                det = det.cpu().numpy()
                boxes_np = det[:, 0:4]
                scores_np = det[:, 4].astype(np.float32)
                class_ids_np = det[:, 5].astype(np.int32)
                
                num_detections = len(boxes_np)

                for i in range(len(boxes_np)):
                    x1, y1, x2, y2 = boxes_np[i]
                    cls_id = int(class_ids_np[i])
                    conf = float(scores_np[i])

                    # Rescale coordinates to original frame size
                    x1n = int(x1 * scale_x)
                    y1n = int(y1 * scale_y)
                    x2n = int(x2 * scale_x)
                    y2n = int(y2 * scale_y)

                    # Use consistent color for each class
                    color = colors[cls_id % len(colors)]
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1n, y1n), (x2n, y2n), color, 2)

                    # Draw label with background
                    label = f"{names[cls_id]} {conf:.2f}"
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    # Label background
                    cv2.rectangle(
                        annotated_frame,
                        (x1n, y1n - label_height - 10),
                        (x1n + label_width, y1n),
                        color,
                        -1
                    )
                    
                    # Label text
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
            draw_fps_info(annotated_frame, fps, avg_fps, max_fps, min_fps)
            draw_detection_info(annotated_frame, num_detections)

            # Display frame
            cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                # Reset FPS statistics
                fps_queue.clear()
                max_fps = 0.0
                min_fps = float("inf")
                print("FPS statistics reset")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")

if __name__ == "__main__":
    main()
