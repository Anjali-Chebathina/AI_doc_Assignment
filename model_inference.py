"""
Robust inference script for LayoutLMv3 + Invoice Field Extraction
"""

import os
import argparse
from io import BytesIO
from PIL import Image
import shutil
import torch
import pytesseract
import re
import json
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification


# -------------------------
# PDF → Image conversion
# -------------------------
def convert_pdf_with_pdf2image(path, dpi=200):
    from pdf2image import convert_from_path
    pages = convert_from_path(path, dpi=dpi)
    return pages[0].convert("RGB")


def convert_pdf_with_fitz(path, dpi=200):
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")


def load_image(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".jpg", ".jpeg", ".png"):
        return Image.open(input_path).convert("RGB")
    if ext == ".pdf":
        try:
            print("📄 Trying pdf2image...")
            return convert_pdf_with_pdf2image(input_path)
        except Exception as e:
            print("⚠️ pdf2image failed, using PyMuPDF (fitz):", e)
            return convert_pdf_with_fitz(input_path)
    raise ValueError(f"Unsupported file type: {ext}")


# -------------------------
# Tesseract detection
# -------------------------
def find_tesseract_executable():
    path_from_which = shutil.which("tesseract")
    if path_from_which:
        return path_from_which
    common = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for p in common:
        if os.path.isfile(p):
            return p
    return None


# -------------------------
# Auto-detect test file
# -------------------------
def autodetect_file(data_dir):
    supported = (".pdf", ".png", ".jpg", ".jpeg")
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(supported)]
    if not files:
        raise FileNotFoundError(f"No supported files in {data_dir}")
    files.sort()
    return os.path.join(data_dir, files[0])


# -------------------------
# Extract key fields from OCR text
# -------------------------
def extract_invoice_fields(text):
    fields = {}

    invoice_no = re.search(r"(?:invoice\s*no\.?|inv\s*#|bill\s*no\.?)[:\s\-]*([A-Z0-9/\-]+)", text, re.I)
    date = re.search(r"(?:date|invoice\s*date)[:\s\-]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})", text, re.I)
    total = re.search(r"(?:total\s*amount|amount\s*due|grand\s*total)[:\s₹]*([\d,]+(?:\.\d{2})?)", text, re.I)
    vendor = re.search(r"(?:from|vendor|seller)[:\s\-]*([A-Za-z0-9 &.,]+)", text, re.I)
    buyer = re.search(r"(?:to|billed\s*to|customer|buyer)[:\s\-]*([A-Za-z0-9 &.,]+)", text, re.I)

    if invoice_no:
        fields["invoice_no"] = invoice_no.group(1).strip()
    if date:
        fields["invoice_date"] = date.group(1).strip()
    if total:
        fields["total_amount"] = "₹" + total.group(1).strip()
    if vendor:
        fields["vendor"] = vendor.group(1).strip()
    if buyer:
        fields["buyer"] = buyer.group(1).strip()

    return fields


# -------------------------
# MAIN
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="../models/layoutlmv3_finetuned")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.normpath(os.path.join(base_dir, args.model_dir))
    data_dir = os.path.normpath(os.path.join(base_dir, args.data_dir))

    print("🔄 Loading LayoutLMv3 processor and model...")
    try:
        processor = LayoutLMv3Processor.from_pretrained(model_dir)
        model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
        print("✅ Loaded local fine-tuned model.")
    except Exception:
        print("⚠️ Local model not found, using microsoft/layoutlmv3-base.")
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

    test_file = os.path.abspath(args.file) if args.file else autodetect_file(data_dir)
    print(f"🖼️ Using file: {test_file}")

    image = load_image(test_file)

    tess = find_tesseract_executable()
    if tess:
        pytesseract.pytesseract.tesseract_cmd = tess
        print(f"✅ Found Tesseract at: {tess}")
        text = pytesseract.image_to_string(image)
    else:
        print("⚠️ No Tesseract found, skipping OCR.")
        text = ""

    encoding = processor(images=image, text=text, return_tensors="pt", truncation=True)
    outputs = model(**encoding)
    print("✅ Inference complete.")

    fields = extract_invoice_fields(text)
    if not fields:
        fields = {"info": "No structured fields extracted (regex failed)."}

    extracted = {
        "document_type": "invoice",
        "fields_extracted": fields
    }

    os.makedirs("../outputs", exist_ok=True)
    out_path = "../outputs/invoice_extracted_fields.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=4, ensure_ascii=False)
    print(f"💾 Saved extracted fields → {out_path}")

    print("\n📋 Extracted Fields:")
    for k, v in fields.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
