import json
import os

def load_extracted_fields(file_path):
    """
    Loads the extracted fields JSON file safely with encoding fallbacks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}\nRun model_inference.py first to generate it.")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        print(f"⚠️ UTF-8 decode failed for {file_path}. Trying Latin-1 encoding...")
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(
                f"❌ The file '{file_path}' seems to contain binary (non-text) data.\n"
                "➡️ Please delete it and rerun model_inference.py to regenerate a valid JSON file."
            )
    except json.JSONDecodeError:
        raise ValueError(f"❌ The file '{file_path}' is empty or invalid JSON. Please regenerate it.")
    
    return data


def reasoning_engine(extracted_data):
    """
    Example reasoning logic to validate and interpret extracted fields.
    """
    if not extracted_data:
        raise ValueError("❌ No data provided for reasoning.")

    doc_type = extracted_data.get("document_type", "unknown").lower()
    fields = extracted_data.get("fields_extracted", {})

    print(f"\n📄 Document Type: {doc_type}")
    print(f"🧩 Extracted Fields: {fields}")

    # Example simple reasoning logic
    if doc_type == "invoice":
        invoice_no = fields.get("invoice_no")
        total_amount = fields.get("total_amount")
        vendor = fields.get("vendor_name")

        if invoice_no and total_amount:
            print("✅ Invoice fields look complete.")
        else:
            print("⚠️ Some key invoice fields missing (invoice_no / total_amount).")

    elif doc_type == "resume":
        name = fields.get("name")
        skills = fields.get("skills", [])
        experience = fields.get("experience")

        if not name:
            print("⚠️ Missing candidate name field.")
        else:
            print(f"✅ Resume appears valid for candidate: {name}")
            if skills:
                print(f"🧠 Skills detected: {', '.join(skills)}")
    
    else:
        print("ℹ️ Unknown document type; no specific reasoning rules applied.")


def main():
    extracted_output = r"C:\Users\anjal\OneDrive\Documents\AI_assignment\outputs\invoice_extracted_fields.json"
    data = load_extracted_fields(extracted_output)
    reasoning_engine(data)


if __name__ == "__main__":
    main()
