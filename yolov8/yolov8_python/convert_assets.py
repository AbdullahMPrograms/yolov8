# convert_assets.py
import numpy as np
import torch

# Load from .npy
anchors_np = np.load("./anchors.npy", allow_pickle=True)
strides_np = np.load("./strides.npy", allow_pickle=True)

# Convert to torch tensors
anchors_pt = torch.tensor(anchors_np)
strides_pt = torch.tensor(strides_np)

# Save as .pt files
torch.save(anchors_pt, "anchors.pt")
torch.save(strides_pt, "strides.pt")

print("Saved anchors.pt and strides.pt successfully.")