from datasets import load_dataset
import json
import random
from collections import defaultdict

# Settings
OUTPUT_FILE = "subset_info.json"
SELECTED_CLASSES = ["racing", "robot", "rubicCupe", "sepia", "shark"]  # your 5 chosen classes
TRAIN_RATIO = 0.8
SEED = 42

def prepare():
    random.seed(SEED)
    print("Loading LaSOT dataset from Hugging Face (streaming mode)...")
    dataset = load_dataset("l-lt/LaSOT", split="train", streaming=True)

    print(f"Filtering only classes: {SELECTED_CLASSES}")
    subset_info = defaultdict(lambda: {"train": [], "test": []})

    # Iterate through dataset stream (no full download)
    for item in dataset:
        cls = item["class"]
        if cls in SELECTED_CLASSES:
            subset_info[cls]["train"].append(item["video_name"])  # each item has metadata like video_name

    # Split train/test
    for cls in SELECTED_CLASSES:
        random.shuffle(subset_info[cls]["train"])
        n = len(subset_info[cls]["train"])
        split_idx = int(n * TRAIN_RATIO)
        subset_info[cls]["test"] = subset_info[cls]["train"][split_idx:]
        subset_info[cls]["train"] = subset_info[cls]["train"][:split_idx]
        subset_info[cls]["train_size"] = len(subset_info[cls]["train"])
        subset_info[cls]["test_size"] = len(subset_info[cls]["test"])

    # Save subset info
    with open(OUTPUT_FILE, "w") as f:
        json.dump(subset_info, f, indent=2)

    print(f"\n Wrote subset info to {OUTPUT_FILE}")
    print(json.dumps(subset_info, indent=2))

if __name__ == "__main__":
    prepare()
