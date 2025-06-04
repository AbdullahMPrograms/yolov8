import time
import torch
import torch.nn as nn
import onnxruntime
import numpy as np
import cv2

from yolov8_utils import *

def frame_process(frame, input_shape=(640, 640)):
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
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim) 
    return torch.cat((x1y1, x2y2), dim)  

def post_process(x):
    dfl = DFL(16)
    anchors = torch.tensor(np.load("./anchors.npy", allow_pickle=True))
    strides = torch.tensor(np.load("./strides.npy", allow_pickle=True))

    box, cls = torch.cat(
        [xi.view(x[0].shape[0], 144, -1) for xi in x], dim=2
    ).split((16*4, 80), dim=1)

    dbox = dist2bbox(dfl(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
    y = torch.cat((dbox, cls.sigmoid()), dim=1)  
    return y, x

# Load COCO names into a list
with open("coco.names", "r") as f:
    names = [n.strip() for n in f.readlines()]

# Create ONNX Runtime session
onnx_model_path = "yolov8m.onnx"
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

inp_meta = npu_session.get_inputs()[0]

# OpenCV window
cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

fps_reported = cap.get(cv2.CAP_PROP_FPS)
print("Driver reports camera FPS =", fps_reported)

prev_time = time.time()
max_fps = 0.0
min_fps = float("inf")

while True:
    # Grab a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Compute FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    max_fps = max(fps, max_fps)
    min_fps = min(fps, min_fps)

    # Preprocess frame
    im = frame_process(frame, input_shape=(640, 640))
    im = im.unsqueeze(0)

    # Convert to NHWC and contiguous float32
    nhwc = im.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    nhwc = np.ascontiguousarray(nhwc)

    # Run inference
    outputs = npu_session.run(
        None,
        {inp_meta.name: nhwc}
    )

    # Convert ONNX outputs Torch tensors [N, C, H, W]
    outputs = [torch.tensor(o).permute(0, 3, 1, 2) for o in outputs]

    # Post‐process
    preds = post_process(outputs)
    preds = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.7, agnostic=False, max_det=300, classes=None)

    det = preds[0] 

    # Prepare arrays for boxes, scores, class IDs
    if det is None or det.numel() == 0:
        boxes_np = np.zeros((0, 4), dtype=np.float32)
        scores_np = np.zeros((0,), dtype=np.float32)
        class_ids_np = np.zeros((0,), dtype=np.int32)
    else:
        det = det.cpu().numpy()
        boxes_np     = det[:, 0:4]                    
        scores_np    = det[:, 4].astype(np.float32)   
        class_ids_np = det[:, 5].astype(np.int32)     

    annotated_frame = frame.copy()
    h0, w0 = frame.shape[:2]
    scale_x, scale_y = w0 / 640, h0 / 640

    # Draw each detection
    if boxes_np.shape[0] > 0:
        for i in range(boxes_np.shape[0]):
            x1, y1, x2, y2 = boxes_np[i]
            cls_id = int(class_ids_np[i])
            conf   = float(scores_np[i])

            # Rescale from 640×640
            x1n = int(x1 * scale_x)
            y1n = int(y1 * scale_y)
            x2n = int(x2 * scale_x)
            y2n = int(y2 * scale_y)

            color = [int(c) for c in (255 * np.random.rand(3,))]
            cv2.rectangle(annotated_frame, (x1n, y1n), (x2n, y2n), color, 2)

            label = f"{names[cls_id]} {conf:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                annotated_frame,
                (x1n, y1n - 20),
                (x1n + t_size[0], y1n),
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

    fps_text = f"FPS: {fps:.2f}"
    min_fps_text = f"MIN FPS: {min_fps:.2f}"
    max_fps_text = f"MAX FPS: {max_fps:.2f}"

    cv2.putText(
        annotated_frame,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )
    cv2.putText(
        annotated_frame,
        min_fps_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )
    cv2.putText(
        annotated_frame,
        max_fps_text,
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Show the annotated frame in real time
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
