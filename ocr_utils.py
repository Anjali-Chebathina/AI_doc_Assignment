# src/ocr_utils.py
import os
from typing import List, Dict
from PIL import Image
import pytesseract
import easyocr
from pdf2image import convert_from_path

# Initialize EasyOCR reader globally
reader = None

def init_ocr():
    """
    Initialize EasyOCR reader once to avoid reloading on every call.
    """
    global reader
    if reader is None:
        print("🔄 Initializing EasyOCR reader...")
        reader = easyocr.Reader(['en'], gpu=False)
    return reader


def extract_text_from_image(image_path: str) -> List[Dict]:
    """
    Extracts text + bounding boxes from an image using EasyOCR.
    Returns a list of dictionaries with text and bounding boxes.
    """
    r = init_ocr()
    results = r.readtext(image_path)
    detections = []
    for bbox, text, conf in results:
        detections.append({
            "text": text,
            "bbox": bbox,
            "confidence": float(conf)
        })
    return detections


def extract_text_from_pdf(pdf_path: str, output_dir="outputs/pdf_pages") -> List[Dict]:
    """
    Converts a PDF into images and runs EasyOCR on each page.
    Returns structured OCR output.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pages = convert_from_path(pdf_path, dpi=300, poppler_path=r"C:\Users\anjal\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin")

    print(f"🧾 Found {len(pages)} pages in {pdf_path}")

    all_pages = []
    for i, page in enumerate(pages):
        img_path = os.path.join(output_dir, f"page_{i+1}.png")
        page.save(img_path, "PNG")
        text_blocks = extract_text_from_image(img_path)
        all_pages.append({
            "page_number": i + 1,
            "text_blocks": text_blocks
        })

    print(f"✅ OCR extraction completed for {len(all_pages)} pages.")
    return all_pages


if __name__ == "__main__":
    # Test with a sample file
    test_path = "data/sample_invoice.pdf"  # put any sample PDF or image here
    if os.path.exists(test_path):
        result = extract_text_from_pdf(test_path)
        print(f"Extracted {sum(len(p['text_blocks']) for p in result)} text regions.")
    else:
        print("⚠️ Please add a 'sample_invoice.pdf' or image file in the data folder to test OCR.")
