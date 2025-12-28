#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights

# -----------------------
# CONFIG (seninle aynı)
# -----------------------
MODEL_PATH = os.path.expanduser("~/ros2_ws_robot/src/lane_inference/model/lane_lraspp_mbv3_ft_best_iou.pth")

IN_DIR  = os.path.expanduser("/home/robot/datasets/bottom_samples_v2")   # değiştir
OUT_DIR = os.path.expanduser("/home/robot/datasets/bottom_samples_out")

IMG_W, IMG_H = 800, 600
NUM_CLASSES = 2
LANE_CLASS_ID = 1

USE_ARGMAX = True
THRESH = 0.50

# "rgb" = BGR->RGB, "bgr" = dokunma
COLOR_MODE = "rgb"

# Pi'nin bgra8 pipeline'ını kabaca simüle etmek istersen:
SIMULATE_BGRA_PIPELINE = False  # True yapıp bir de öyle dene

# overlay ayarı
MASK_COLOR = (0, 255, 0)  # BGR
ALPHA = 0.45


def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.resize(img_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

    if COLOR_MODE == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()
    return t


def simulate_bgra_pipeline(img_bgr: np.ndarray) -> np.ndarray:
    # BGR -> BGRA -> (ROS'dan gelen gibi) tekrar BGR
    a = np.full((img_bgr.shape[0], img_bgr.shape[1], 1), 255, dtype=np.uint8)
    bgra = np.concatenate([img_bgr, a], axis=2)
    bgr2 = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    return bgr2


def overlay_mask(img_bgr: np.ndarray, mask255: np.ndarray) -> np.ndarray:
    # mask255: 0/255 mono
    out = img_bgr.copy()
    colored = np.zeros_like(out)
    colored[mask255 > 0] = MASK_COLOR
    out = cv2.addWeighted(out, 1.0, colored, ALPHA, 0)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cpu")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)

    model = lraspp_mobilenet_v3_large(
        weights=None,
        weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
        num_classes=NUM_CLASSES,
    ).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths += glob.glob(os.path.join(IN_DIR, ext))
    paths = sorted(paths)

    if not paths:
        print("No images found in:", IN_DIR)
        return

    print("Found:", len(paths), "images")
    print("COLOR_MODE:", COLOR_MODE, "| SIMULATE_BGRA_PIPELINE:", SIMULATE_BGRA_PIPELINE)

    with torch.inference_mode():
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue

            if SIMULATE_BGRA_PIPELINE:
                img = simulate_bgra_pipeline(img)

            inp = preprocess(img).to(device)

            logits = model(inp)["out"]  # 1xCxhxw
            if logits.shape[-2] != IMG_H or logits.shape[-1] != IMG_W:
                logits = F.interpolate(logits, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)

            if USE_ARGMAX:
                pred = torch.argmax(logits, dim=1)[0].to(torch.uint8).cpu().numpy()
                mask = (pred == LANE_CLASS_ID).astype(np.uint8) * 255
            else:
                probs = torch.softmax(logits, dim=1)[0, LANE_CLASS_ID].cpu().numpy()
                mask = (probs >= THRESH).astype(np.uint8) * 255

            ov = overlay_mask(cv2.resize(img, (IMG_W, IMG_H)), mask)

            base = os.path.splitext(os.path.basename(p))[0]
            cv2.imwrite(os.path.join(OUT_DIR, f"{base}_mask.png"), mask)
            cv2.imwrite(os.path.join(OUT_DIR, f"{base}_overlay.jpg"), ov, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()