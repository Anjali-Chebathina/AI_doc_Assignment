import json
import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, Trainer, TrainingArguments
from datasets import load_dataset

print("🔄 Loading LayoutLMv3 processor and dataset...")

# ✅ FIX: Disable auto OCR since dataset already includes bounding boxes
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

# Load limited samples for faster testing
dataset = load_dataset("naver-clova-ix/cord-v2", split="train[:50]")

# ✅ Label definitions
labels = [
    "O", "B-company", "I-company", "B-date", "I-date",
    "B-total", "I-total", "B-address", "I-address", "B-item", "I-item"
]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}


# ✅ Preprocessing function
def preprocess(example):
    image = example["image"]

    # Detect ground truth field
    gt = None
    for key in ["gt_parse", "ground_truth", "annotations"]:
        if key in example:
            gt = example[key]
            break

    if gt is None:
        print("⚠️ Skipping example: No ground truth key found.")
        return {}

    # Parse JSON string if needed
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except Exception as e:
            print(f"⚠️ Failed to parse JSON: {e}")
            return {}

    # Extract words, boxes, and labels
    words, boxes, word_labels = [], [], []
    lines = gt.get("valid_line", gt.get("lines", []))

    for line in lines:
        for word_info in line.get("words", []):
            word = word_info.get("text", "").strip()
            box = word_info.get("box", [0, 0, 0, 0])
            if word:
                words.append(word)
                boxes.append(box)
                word_labels.append(label2id["O"])

    if not words:
        return {}

    # Encode using processor (we provide boxes manually)
    encoding = processor(
        images=image,
        text=words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    encoding["labels"] = torch.tensor(
        word_labels + [label2id["O"]] * (512 - len(word_labels))
    )[:512]

    return {k: v.squeeze() for k, v in encoding.items()}


print("🧩 Preprocessing dataset...")
processed_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
processed_dataset = processed_dataset.filter(lambda x: "input_ids" in x)
print(f"✅ Successfully preprocessed {len(processed_dataset)} samples!")

# ✅ Initialize model
print("🧠 Initializing LayoutLMv3 model...")
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# ✅ Training configuration
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=5,
    save_total_limit=1,
)

# ✅ Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
)

print("🚀 Starting fine-tuning...")
trainer.train()
print("✅ Training complete! Model saved to ./outputs/")
