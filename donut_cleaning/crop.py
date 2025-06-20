import os
import json
from PIL import Image
from tqdm import tqdm

# Input folders
IMAGES_DIR = './DEHADO-AI_TRAINING_DATASET_COMPLETE/IMAGES_1500'
LABELS_DIR = './DEHADO-AI_TRAINING_DATASET_COMPLETE/LABELS_1500'

# Output folders
OUTPUT_DIR = 'dehado_cropped_dataset'
CROPPED_IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
CROPPED_LABELS_DIR = os.path.join(OUTPUT_DIR, 'labels')

os.makedirs(CROPPED_IMAGES_DIR, exist_ok=True)
os.makedirs(CROPPED_LABELS_DIR, exist_ok=True)

# Precompute which base_names are already done:
done_bases = {
    fname.split('_', 1)[0]
    for fname in os.listdir(CROPPED_IMAGES_DIR)
    if fname.endswith('.jpg')
}

for label_file in tqdm(sorted(os.listdir(LABELS_DIR)), desc="Processing labels"):
    if not label_file.lower().endswith('.json'):
        continue

    base_name = os.path.splitext(label_file)[0]
    if base_name in done_bases:
        continue

    label_path = os.path.join(LABELS_DIR, label_file)
    image_path = os.path.join(IMAGES_DIR, base_name + '.jpg')
    if not os.path.isfile(image_path):
        print(f"[ERROR] No image for {base_name}: expected {image_path}")
        continue

    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            fields = json.load(f)
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in {label_file}, skipping.")
        continue

    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"[ERROR] Cannot open image {image_path}: {e}")
        continue

    for idx, field in enumerate(fields):
        coords = field.get("Coordinate") or field.get("coordinate")
        if not coords or len(coords) != 4:
            # silently skip bad coords
            continue

        x1, y1, x2, y2 = coords
        crop = img.crop((x1, y1, x2, y2))

        name = field.get("Field name", f"field_{idx}")
        safe = name.replace('/', '_').replace(' ', '_')
        img_fname = f"{base_name}_{safe}.jpg"
        lbl_fname = f"{base_name}_{safe}.txt"

        out_img = os.path.join(CROPPED_IMAGES_DIR, img_fname)
        out_lbl = os.path.join(CROPPED_LABELS_DIR, lbl_fname)
        if os.path.exists(out_img):
            continue  # skip already-cropped field

        try:
            crop.save(out_img)
            with open(out_lbl, 'w', encoding='utf-8') as w:
                w.write(field.get("Field value", ""))
        except Exception as e:
            print(f"[ERROR] Writing {img_fname}/{lbl_fname}: {e}")

    # mark this base as done so we don’t re-scan it
    done_bases.add(base_name)
