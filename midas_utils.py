import torch
import cv2
import numpy as np

# Load MiDaS small model with bundled weights
midas = torch.hub.load(
    "intel-isl/MiDaS",
    "MiDaS_small",
    pretrained=True,
    trust_repo=True,
)
midas.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms",
    trust_repo=True,
)
transform = midas_transforms.small_transform


def get_depth_map_bgr(bgr_image: np.ndarray) -> np.ndarray:
    """
    Compute a normalized depth map (0–1) for a BGR image.
    """
    img_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # For MiDaS_small, transform returns a 4D batch tensor [1, 3, H, W]
    inp = transform(img_rgb).to(device)

    with torch.no_grad():
        pred = midas(inp)  # [1, 256, H', W']
        depth = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return depth.astype("float32")
