"""Shared inference pipeline used by FastAPI backend and Streamlit local mode."""

import base64
import logging
import os
import urllib.request
from io import BytesIO
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from model import UNetResNet18

logger = logging.getLogger("polyp_inference")

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "best.pth")
CHECKPOINT_URL = os.environ.get("CHECKPOINT_URL", "")
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_CONTOUR_AREA = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: UNetResNet18 | None = None


def ensure_checkpoint() -> None:
    """Use local checkpoint or download from CHECKPOINT_URL."""
    if os.path.isfile(CHECKPOINT_PATH):
        return

    if not CHECKPOINT_URL:
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Place best.pth locally or set CHECKPOINT_URL to a direct download link."
        )

    logger.info("Downloading checkpoint from %s", CHECKPOINT_URL)
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
    logger.info("Checkpoint saved to %s", CHECKPOINT_PATH)


def load_model() -> UNetResNet18:
    """Load and cache the segmentation model."""
    global _model
    if _model is not None:
        return _model

    ensure_checkpoint()

    net = UNetResNet18(out_channels=1, pretrained=False).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    net.load_state_dict(state_dict)
    net.eval()
    _model = net
    logger.info("Model loaded from %s on %s", CHECKPOINT_PATH, device)
    return _model


def preprocess(image_np: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """Resize to 224x224 and apply ImageNet normalization."""
    resized = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor.to(device), resized


def encode_image_b64(img_array: np.ndarray) -> str:
    """Encode a numpy image as a base64 PNG string."""
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

    if img_array.ndim == 2:
        pil_img = Image.fromarray(img_array, mode="L")
    else:
        pil_img = Image.fromarray(img_array)

    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_inference(image_np: np.ndarray) -> dict[str, Any]:
    """Full inference pipeline: preprocess -> predict -> postprocess."""
    model = load_model()
    tensor, resized_img = preprocess(image_np)

    with torch.no_grad():
        pred = model(tensor)
        confidence = float(pred.max().item())
        mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    bboxes = []
    bbox_img = resized_img.copy()

    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        valid_contours.append(contour)
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    contour_img = resized_img.copy()
    if valid_contours:
        cv2.drawContours(contour_img, valid_contours, -1, (0, 255, 0), 2)

    overlay = resized_img.copy().astype(np.float32) / 255.0
    red_layer = np.zeros_like(overlay)
    red_layer[..., 0] = 1.0
    mask_bool = mask > 0
    alpha = 0.4
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * red_layer[mask_bool]

    return {
        "tumor_present": len(bboxes) > 0,
        "confidence": round(confidence, 4),
        "mask_area_px": int((mask > 0).sum()),
        "num_regions": len(bboxes),
        "bboxes": bboxes,
        "original_image": encode_image_b64(resized_img),
        "mask_image": encode_image_b64(mask),
        "overlay_image": encode_image_b64(overlay),
        "contour_image": encode_image_b64(contour_img),
        "bbox_image": encode_image_b64(bbox_img),
    }
