# convert_assets_to_binary.py
import numpy as np

# Load from .npy
print("Loading anchors.npy and strides.npy...")
anchors_np = np.load("./anchors.npy", allow_pickle=True)
strides_np = np.load("./strides.npy", allow_pickle=True)

# Ensure they are float32, as expected by the C++ code
anchors_np = anchors_np.astype(np.float32)
strides_np = strides_np.astype(np.float32)

# Save as raw binary files
with open("anchors.bin", "wb") as f:
    f.write(anchors_np.tobytes())
print(f"Saved anchors.bin with shape {anchors_np.shape} and dtype {anchors_np.dtype}")

with open("strides.bin", "wb") as f:
    f.write(strides_np.tobytes())
print(f"Saved strides.bin with shape {strides_np.shape} and dtype {strides_np.dtype}")