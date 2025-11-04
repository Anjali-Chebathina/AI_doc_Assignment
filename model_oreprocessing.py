import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import LayoutLMv3Processor
from PIL import Image

print("🔄 Initializing LayoutLMv3 processor...")
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

# Load a few samples
print("🧩 Preprocessing a few samples from CORD-v2...")
dataset = load_dataset("naver-clova-ix/cord-v2", split="train[:10]")

preprocessed_samples = []

for example in tqdm(dataset):
    # 🔹 CORD dataset stores image as an actual image object
    image = example["image"]  # directly access it
    if not isinstance(image, Image.Image):
        # Some versions store a path instead — handle both cases
        image = Image.open(image).convert("RGB")

    words = []
    boxes = []

    gt = example.get("gt_parse", {})
    if not gt:
        continue

    for line in gt.get("valid_line", []):
        for word_info in line.get("words", []):
            if "text" in word_info:
                words.append(str(word_info["text"]))
                box = word_info.get("box", [0, 0, 0, 0])
                if isinstance(box[0], list):
                    box = box[0]
                boxes.append([int(x) for x in box])

    if not words or not boxes:
        continue

    encoded_inputs = processor(
        images=image,
        text=words,
        boxes=boxes,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    preprocessed_samples.append(encoded_inputs)

print(f"✅ Successfully preprocessed {len(preprocessed_samples)} samples!")
