import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from PIL import Image
from ocr_utils import run_ocr
import fitz  # PyMuPDF
import io
import os
import tempfile
from utils import extract_fields_from_ocr, make_decision

# ======================================
# 🔧 Global Model Initialization
# ======================================
MODEL_DIR = os.path.join("models", "layoutlmv3-base")
os.makedirs(MODEL_DIR, exist_ok=True)

print("🔄 Loading LayoutLMv3 model...")

# Load model and processor once
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    cache_dir=MODEL_DIR,
    apply_ocr=False
)
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    cache_dir=MODEL_DIR,
    num_labels=3
)
model.eval()

print("✅ Model loaded successfully!")


# ======================================
# 🧩 Helper Functions
# ======================================
def pdf_to_image(pdf_path):
    """Convert the first page of a PDF to a PIL Image."""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        return img
    except Exception as e:
        raise RuntimeError(f"Error converting PDF to image: {e}")


# Mapping class index → document type
LABEL_MAP = {0: "invoice", 1: "resume", 2: "report"}


# ======================================
# 🚀 Inference Function
# ======================================
def inference_single(file_path):
    """Perform OCR and classification on a PDF or image file.

    Returns:
        dict: {
            doc_label,
            doc_confidence,
            ocr,
            fields_extracted,
            decision
        }
    """
    ext = os.path.splitext(file_path)[1].lower()

    # Convert PDF to image if needed
    if ext == ".pdf":
        image = pdf_to_image(file_path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            temp_image_path = tmp.name
        ocr_path = temp_image_path
    else:
        image = Image.open(file_path).convert("RGB")
        ocr_path = file_path

    # Run OCR
    ocr = run_ocr(ocr_path)

    # Delete temporary image
    if ext == ".pdf" and os.path.exists(ocr_path):
        os.remove(ocr_path)

    # Prepare model input
    words = [o["text"] for o in ocr]
    boxes = [o["box"] for o in ocr]

    encoding = processor(image, words, boxes=boxes, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=-1)
        label_id = torch.argmax(probs, dim=-1).item()
        confidence = float(probs[0, label_id])

    # Extract fields & make decision
    fields = extract_fields_from_ocr(ocr)
    decision = make_decision(confidence, fields)

    return {
        "doc_label": LABEL_MAP.get(label_id, "unknown"),
        "doc_confidence": round(confidence, 4),
        "ocr": ocr,
        "fields_extracted": fields,
        "decision": decision,
    }
