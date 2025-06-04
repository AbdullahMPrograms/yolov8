import torch
import torch.nn as nn
import onnxruntime
import numpy as np
import cv2

from huggingface_hub import hf_hub_download
from yolov8_utils import *

current_dir = get_directories()

# Download Yolov8 model from Ryzen AI model zoo. Registration is required before download.
hf_hub_download(repo_id="amd/yolov8m", filename="yolov8m.onnx", local_dir=str(current_dir))

# display videos in notebook
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output, display, HTML


def frame_process(frame, input_shape=(640, 640)):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_shape)
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    img = np.transpose(img, (2, 0, 1))
    return img
    

class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(
            b, 4, a
        )


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def post_process(x):
    dfl = DFL(16)
    anchors = torch.tensor(
        np.load(
            "./anchors.npy",
            allow_pickle=True,
        )
    )
    strides = torch.tensor(
        np.load(
            "./strides.npy",
            allow_pickle=True,
        )
    )
    box, cls = torch.cat([xi.view(x[0].shape[0], 144, -1) for xi in x], 2).split(
        (16 * 4, 80), 1
    )
    dbox = dist2bbox(dfl(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
    y = torch.cat((dbox, cls.sigmoid()), 1)
    return y, x

# Load labels of coco dataaset
with open('coco.names', 'r') as f:
        names = f.read()

imgsz = [640, 640]

# Specify the path to the quantized ONNZ Model
onnx_model_path = "yolov8m.onnx"

npu_options = onnxruntime.SessionOptions()

# Load the JSON from disk
with open("./vaip_config.json", "r", encoding="utf-8") as f:
    config_text = f.read()

# Build provider_options using the actual JSON string
provider_options = [{
    "config": config_text,
    "ai_analyzer_visualization": True,
    "ai_analyzer_profiling": True,
}]

npu_session = onnxruntime.InferenceSession(
    onnx_model_path,
    providers = ['VitisAIExecutionProvider'],
    sess_options=npu_options,
    provider_options = provider_options
)

with open('coco.names', 'r') as f:
        names = f.read()

# Video input
cap = cv2.VideoCapture(0)

while (True):
    try:
        clear_output(wait=True)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        input_shape = (640, 640)

        im = frame_process(frame, input_shape)
        if len(im.shape) == 3:
            im = im[None]
        
        # outputs = npu_session.run(None, {npu_session.get_inputs()[0].name: im.permute(0, 2, 3, 1).cpu().numpy()})
        
        # Transpose to NHWC
        nhwc_torch = im.permute(0, 2, 3, 1)  # shape [1, 640, 640, 3]
        # Convert to NumPy
        nhwc_np = nhwc_torch.cpu().numpy()
        # Force it to be a C-contiguous float32 array
        nhwc_contig = np.ascontiguousarray(nhwc_np, dtype=np.float32)
        # Feed into ONNX Runtime
        outputs = npu_session.run(None,{ npu_session.get_inputs()[0].name: nhwc_contig })

        # Postprocessing
        outputs = [torch.tensor(item).permute(0, 3, 1, 2) for item in outputs]
        preds = post_process(outputs)
        preds = non_max_suppression(
            preds, 0.25, 0.7, agnostic=False, max_det=300, classes=None
        )

        colors = [[random.randint(0, 255) for _ in range(3)] 
                for _ in range(len(names))]

        plot_images(
        im,
        *output_to_target(preds, max_det=15),
        frame,
        fname="output.jpg",
        names=names,
        )
        
    except KeyboardInterrupt:
        cap.release()



