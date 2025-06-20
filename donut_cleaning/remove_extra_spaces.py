import os
import json
import re
from glob import glob
from tqdm import tqdm

# ====== CONFIG ======
LABELS_DIR = "./DEHADO-AI_TRAINING_DATASET_COMPLETE/LABELS_1500"
# ====================

def clean_whitespace_in_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed = False
    for entry in data:
        if "Field value" in entry and isinstance(entry["Field value"], str):
            orig = entry["Field value"]
            # collapse all whitespace (spaces, tabs, newlines) to a single space, then strip ends
            cleaned = re.sub(r"\s+", " ", orig).strip()
            if cleaned != orig:
                entry["Field value"] = cleaned
                changed = True

    if changed:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    return changed

def main():
    json_files = glob(os.path.join(LABELS_DIR, "*.json"))
    modified_count = 0

    print(f"Found {len(json_files)} label files. Cleaning whitespace…")
    for jp in tqdm(json_files, desc="Cleaning JSONs"):
        if clean_whitespace_in_json(jp):
            modified_count += 1

    print(f"Done. Modified {modified_count} out of {len(json_files)} files.")

if __name__ == "__main__":
    main()
