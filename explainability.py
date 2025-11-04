# src/explainability.py
"""
Explainability helper for LayoutLMv3 (robust).
- Auto-detects an image in ../data (first supported file).
- Detects Tesseract availability and falls back to apply_ocr=False if not found.
- Loads local fine-tuned model if available, otherwise falls back to microsoft/layoutlmv3-base.
- Runs inference with output_attentions=True and displays a simple attention heatmap (if available).
"""

import os
import shutil
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# -------------------------
# Helpers
# -------------------------
def find_tesseract_executable():
    """Return path to tesseract if available on PATH or common Windows locations, else None."""
    # 1) shutil.which
    t = shutil.which("tesseract")
    if t:
        return t

    # 2) common Windows install locations
    common = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for p in common:
        if os.path.isfile(p):
            return p
    return None

def autodetect_image(data_dir):
    """Return first supported image/pdf file in data_dir (jpg/png/jpeg/pdf) or None."""
    supported = ("*.jpg", "*.jpeg", "*.png", "*.pdf")
    for pat in supported:
        matches = sorted(glob.glob(os.path.join(data_dir, pat)))
        if matches:
            return matches[0]
    return None

def show_image_and_heatmap(image_pil, heatmap, title="Attention heatmap"):
    """
    Show original image and heatmap side-by-side.
    heatmap expected as a 2D numpy array. We'll resize heatmap to image size for visual effect.
    """
    if heatmap is None:
        plt.imshow(image_pil)
        plt.title("Image (no attention available)")
        plt.axis("off")
        plt.show()
        return

    # normalize heatmap
    hm = np.array(heatmap, dtype=float)
    if hm.size == 0:
        plt.imshow(image_pil); plt.title("Image (empty attention)"); plt.axis("off"); plt.show(); return
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-12)

    # resize heatmap to image size using PIL (nearest)
    img_w, img_h = image_pil.size
    # convert heatmap to PIL image
    hm_img = Image.fromarray(np.uint8(hm * 255))
    hm_img = hm_img.resize((img_w, img_h), resample=Image.BILINEAR)

    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].imshow(image_pil)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(image_pil)
    axs[1].imshow(hm_img, cmap="jet", alpha=0.5)  # overlay
    axs[1].set_title(title)
    axs
