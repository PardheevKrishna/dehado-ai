import os
import math
import random
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel
from jiwer import wer
from tqdm import tqdm

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DATA_ROOT     = "../donut_train/dehado_cropped_dataset"
IMAGES_DIR    = os.path.join(DATA_ROOT, "images")
LABELS_DIR    = os.path.join(DATA_ROOT, "labels")
MODEL_NAME    = "naver-clova-ix/donut-base"
CHECKPOINT    = "../donut_train/donut_best_checkpoint.pt"
MAX_LENGTH    = 256
BATCH_SIZE    = 4
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED          = 42

# -------------------------------------------------------------------
# SEED
# -------------------------------------------------------------------
torch.manual_seed(SEED)
random.seed(SEED)

# -------------------------------------------------------------------
# DATASET & HELPERS (inlined)
# -------------------------------------------------------------------
class DonutDataset(Dataset):
    def __init__(self, images_dir, labels_dir, processor, max_length=MAX_LENGTH):
        self.image_paths = sorted(glob(os.path.join(images_dir, '*')))
        self.label_paths = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob(os.path.join(labels_dir, '*.txt'))
        }
        self.processor  = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        key        = os.path.splitext(os.path.basename(image_path))[0]
        label_path = self.label_paths.get(key)
        with Image.open(image_path).convert("RGB") as img, \
             open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip().lower()

        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        return pixel_values, labels, text, key, image_path

def compute_cer(preds, gts):
    total_edits, total_chars = 0, 0
    for p, g in zip(preds, gts):
        edits = wer(g, p) * len(g.split())
        total_edits  += edits
        total_chars  += len(g)
    return total_edits / total_chars if total_chars > 0 else 0.0

def compute_field_and_document_accuracy(preds, gts, keys):
    # field‐level
    correct_fields = sum(1 for p, g in zip(preds, gts) if p.strip() == g.strip())
    total_fields   = len(gts)
    field_acc = correct_fields / total_fields if total_fields > 0 else 0.0

    # document‐level, **fixed grouping**
    doc_groups = {}
    for key, p, g in zip(keys, preds, gts):
        parts  = key.split('_')
        doc_id = '_'.join(parts[:2])            # ← take only MIT_1, MIT_34, etc.
        doc_groups.setdefault(doc_id, []).append((p.strip(), g.strip()))

    correct_docs = sum(1 for pairs in doc_groups.values() if all(p == g for p, g in pairs))
    total_docs   = len(doc_groups)
    doc_acc = correct_docs / total_docs if total_docs > 0 else 0.0

    return field_acc, doc_acc

# -------------------------------------------------------------------
# LOAD PROCESSOR & MODEL
# -------------------------------------------------------------------
processor = DonutProcessor.from_pretrained(MODEL_NAME)
processor.image_processor.size = {"height":512, "width":512}

model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.config.max_length             = MAX_LENGTH
model.config.pad_token_id           = processor.tokenizer.pad_token_id
model.config.eos_token_id           = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.eos_token)
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.cls_token)
model.to(DEVICE)

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])

# -------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------
dataset = DonutDataset(IMAGES_DIR, LABELS_DIR, processor, max_length=MAX_LENGTH)
loader  = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda batch: (
        torch.stack([item[0] for item in batch]).to(DEVICE),
        None,
        [item[2] for item in batch],
        [item[3] for item in batch],
        [item[4] for item in batch],
    )
)

# -------------------------------------------------------------------
# INFERENCE
# -------------------------------------------------------------------
all_preds, all_texts, all_keys, all_image_paths = [], [], [], []
model.eval()
print("Running inference…")
for pixel_values, _, texts, keys, image_paths in tqdm(loader, desc="Batches"):
    with torch.no_grad():
        gen_ids = model.generate(
            pixel_values,
            max_length=MAX_LENGTH,
            num_beams=1,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.eos_token),
        )
        decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)

    all_preds.extend(decoded)
    all_texts.extend(texts)
    all_keys.extend(keys)
    all_image_paths.extend(image_paths)

# -------------------------------------------------------------------
# METRICS & COUNTS
# -------------------------------------------------------------------
cer_pct       = compute_cer(all_preds, all_texts) * 100
wer_pct       = wer(" ".join(all_texts), " ".join(all_preds)) * 100
field_acc_frac, doc_acc_frac = compute_field_and_document_accuracy(all_preds, all_texts, all_keys)
field_acc_pct = field_acc_frac * 100
doc_acc_pct   = doc_acc_frac * 100

total_fields   = len(all_texts)
correct_fields = int(field_acc_frac * total_fields)
wrong_fields   = total_fields - correct_fields

# Re-compute document grouping *with the same logic* to get counts
doc_groups = {}
for key, p, g in zip(all_keys, all_preds, all_texts):
    parts  = key.split('_')
    doc_id = '_'.join(parts[:2])
    doc_groups.setdefault(doc_id, []).append((p.strip(), g.strip()))

total_docs   = len(doc_groups)
correct_docs = sum(1 for pairs in doc_groups.values() if all(p==g for p,g in pairs))
wrong_docs   = total_docs - correct_docs

# final score
final_score = (
    0.35 * (100 - wer_pct) +
    0.35 * (100 - cer_pct) +
    0.15 * field_acc_pct +
    0.15 * doc_acc_pct
)

mismatch_idxs = [i for i, (p, g) in enumerate(zip(all_preds, all_texts)) if p.strip() != g.strip()]

# -------------------------------------------------------------------
# PRINT SUMMARY
# -------------------------------------------------------------------
print("\n" + "="*60)
print("TEST SET PERFORMANCE")
print(f"  CER:                {cer_pct:.2f}%")
print(f"  WER:                {wer_pct:.2f}%")
print(f"  Field Acc:          {field_acc_pct:.2f}%  ({correct_fields}/{total_fields} correct, {wrong_fields} wrong)")
print(f"  Document Acc:       {doc_acc_pct:.2f}%  ({correct_docs}/{total_docs} correct, {wrong_docs} wrong)")
print(f"  Final Score:        {final_score:.2f}")
print("="*60)

# -------------------------------------------------------------------
# STEP THROUGH MISMATCHES
# -------------------------------------------------------------------
print(f"\nTotal mismatches: {len(mismatch_idxs)}/{total_fields} fields.")
print("Press Enter to open each image+label; Ctrl+C to quit early.\n")

for idx in mismatch_idxs:
    img_path   = all_image_paths[idx]
    label_path = os.path.join(LABELS_DIR, f"{all_keys[idx]}.txt")

    print(f"[{idx}] Key={all_keys[idx]}")
    print(f"  GT:   {all_texts[idx]}")
    print(f"  Pred: {all_preds[idx]}")
    print(f"Opening:\n    {img_path}\n    {label_path}")
    os.startfile(img_path)
    os.startfile(label_path)
    input("→ Press Enter to continue…")

print("\nAll done!")
