import torch
import cv2
import numpy as np

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
    img_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # transform() for MiDaS_small already returns a 4D batch tensor [1, 3, H, W]
    inp = transform(img_rgb).to(device)  # shape: [1, 3, h, w]

    with torch.no_grad():
        pred = midas(inp)  # do NOT unsqueeze again
        depth = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return depth.astype("float32")
