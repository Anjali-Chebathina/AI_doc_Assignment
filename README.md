# 🧠 End-to-End AI System for Intelligent Document Understanding and Automated Decision-Making  
**Company:** Flikt Technology Web Solutions  
**Project Type:** AI Developer Training Assignment  
**Author:** Anjali Chebathina  
**Date:** November 2025  

---

## 🚀 Overview

This project focuses on building an **End-to-End AI System** capable of automatically **extracting, understanding, and making decisions** based on **unstructured business documents** (invoices, resumes, and reports).

It integrates **OCR**, **Document Layout Understanding**, **Deep Learning**, and **Reasoning Layers** to enable intelligent automation for business workflows.

---

## 🏗️ System Architecture

📂 AI_ASSIGNMENT
│
├── 📄 sample_invoice.pdf
├── 📄 resume_data.csv
│
├── 📁 data/
│ └── sample_invoice.jpg
│
├── 📁 models/
│ └── layoutlmv3_finetuned/
│
├── 📁 outputs/
│ ├── checkpoint-50/
│ └── pdf_pages/
│
├── 📁 src/
│ ├── data_loader.py
│ ├── model_oreprocessing.py
│ ├── model_training.py
│ ├── model_inference.py
│ └── ocr_utils.py
│
└── 📁 venv/

---

## 🧩 Key Components

| Module | Description |
|--------|--------------|
| **`data_loader.py`** | Handles dataset import, OCR extraction from PDFs and images. |
| **`ocr_utils.py`** | Includes text extraction functions using EasyOCR, Tesseract, and PyMuPDF. |
| **`model_oreprocessing.py`** | Prepares document images, bounding boxes, and annotations for model input. |
| **`model_training.py`** | Fine-tunes **LayoutLMv3** on structured document data (CORD-v2 format). |
| **`model_inference.py`** | Runs inference using fine-tuned model to extract structured fields from unseen documents. |

---

## 🧠 Model Architecture

- **Base Model:** `LayoutLMv3` (Microsoft)
- **Processor:** OCR + LayoutLMv3 Processor  
- **Training Dataset:** CORD-v2 / Custom structured document data  
- **Frameworks:** PyTorch, Hugging Face Transformers  
- **Key Outputs:**  
  - Extracted entities (invoice number, date, total, etc.)  
  - Structured JSON representation of document fields  

---

## ⚙️ Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Anjali-Chebathina/AI_doc_Assignment.git
cd AI_ASSIGNMENT
2️⃣ Set Up Virtual Environment
python -m venv venv
venv\Scripts\activate   # (Windows)

3️⃣ Install Dependencies
pip install -r requirements.txt

🧾 Usage Instructions
Step 1: Data Loading

Run the OCR and dataset loading scripts:

python src/data_loader.py

Step 2: Model Preprocessing
python src/model_oreprocessing.py

Step 3: Model Training
python src/model_training.py

Step 4: Model Inference
python src/model_inference.py


🖼️ Screenshots

📸  screenshots below to show UI, training logs, and results
<img width="1131" height="837" alt="Screenshot 2025-11-04 202920" src="https://github.com/user-attachments/assets/357890ac-f339-4cb4-85ad-2a0eb3def8da" />


	Data preprocessing preview

	Model training in progress

	Final inference output visualization
<img width="1139" height="839" alt="Screenshot 2025-11-04 203013" src="https://github.com/user-attachments/assets/3f88c53d-7f83-479d-9fed-07a71d9bd129" />

🧠 Future Enhancements

Integrate Donut / LayoutLMv3-Large for better multimodal reasoning

Build FastAPI-based REST API for deployment

Add explainability layers (attention maps / SHAP)

Extend reasoning to decision-making modules (invoice validation, resume ranking)

🏢 About Flikt Technology Web Solutions

Flikt Technology Web Solutions is a digital innovation company focused on AI-driven automation and web solutions.
This project is part of an AI Developer Training assignment designed to evaluate real-world ML & AI system design capabilities.

📜 License

This project is for educational and research purposes only.
All model weights and data used belong to their respective owners.

👩‍💻 Author

Anjali Chebathina
AI Developer Intern — Flikt Technology Web Solutions
📧 [Add your email here]

⭐ Acknowledgments

Microsoft Research for LayoutLMv3

Hugging Face for Transformers

CORD-v2 dataset contributors

EasyOCR and PyMuPDF for OCR utilities
