import os
import random
import math
import torch
import torch.optim as optim
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from tqdm import tqdm
from jiwer import wer
from transformers import DonutProcessor, VisionEncoderDecoderModel, get_linear_schedule_with_warmup
from torch.amp import GradScaler
from torchvision.utils import make_grid

# -------------------------------------------------------------------
# PATHS & HYPERPARAMETERS
# -------------------------------------------------------------------
DATA_ROOT         = "./dehado_cropped_dataset"
IMAGES_DIR        = os.path.join(DATA_ROOT, "images")
LABELS_DIR        = os.path.join(DATA_ROOT, "labels")
MODEL_NAME        = "naver-clova-ix/donut-base"
BATCH_SIZE        = 4    
ACCUM_STEPS       = 2    
LR                = 5e-5
EPOCHS            = 1000
MAX_LENGTH        = 256
CHECKPOINT        = "donut_checkpoint.pt"
BEST_CHECKPOINT   = "donut_best_checkpoint.pt"
DEVICE            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED              = 42
LOG_DIR           = "runs/donut_experiment"
IMAGE_LOG_FREQ    = 100  

# -------------------------------------------------------------------
# SET SEED
# -------------------------------------------------------------------
torch.manual_seed(SEED)
random.seed(SEED)

# -------------------------------------------------------------------
# PROCESSOR & MODEL
# -------------------------------------------------------------------
processor = DonutProcessor.from_pretrained(MODEL_NAME)
processor.image_processor.size = {"height":512, "width":512}

model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.config.max_length               = MAX_LENGTH
model.config.pad_token_id             = processor.tokenizer.pad_token_id
model.config.eos_token_id             = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.eos_token)
model.config.decoder_start_token_id   = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.cls_token)
model.config.forced_eos_token_id      = model.config.eos_token_id

# Freeze half of encoder layers to save memory
for i, layer in enumerate(model.encoder.encoder.layers):
    if i < len(model.encoder.encoder.layers) // 2:
        for param in layer.parameters():
            param.requires_grad = False

# -------------------------------------------------------------------
# DATASET
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
        return pixel_values, labels, text, key

def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch]).to(DEVICE)
    labels       = torch.stack([item[1] for item in batch]).to(DEVICE)
    texts        = [item[2] for item in batch]
    keys         = [item[3] for item in batch]
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return pixel_values, labels, texts, keys

def compute_cer(preds, gts):
    total_edits, total_chars = 0, 0
    for p, g in zip(preds, gts):
        edits = wer(g, p) * len(g.split())
        total_edits  += edits
        total_chars  += len(g)
    return total_edits / total_chars if total_chars > 0 else 0.0

def compute_field_and_document_stats(preds, gts, keys):
    # Field-level
    correct_fields = sum(1 for p, g in zip(preds, gts) if p.strip() == g.strip())
    total_fields   = len(gts)
    wrong_fields   = total_fields - correct_fields

    # Document-level
    doc_groups = {}
    for key, p, g in zip(keys, preds, gts):
        doc_id = key.rsplit('_', 1)[0]
        doc_groups.setdefault(doc_id, []).append((p.strip(), g.strip()))
    correct_docs = sum(1 for pairs in doc_groups.values() if all(p == g for p, g in pairs))
    total_docs   = len(doc_groups)
    wrong_docs   = total_docs - correct_docs

    return (correct_fields, wrong_fields, total_fields,
            correct_docs, wrong_docs, total_docs)

# -------------------------------------------------------------------
# MAIN TRAINING LOOP
# -------------------------------------------------------------------
def main():
    dataset    = DonutDataset(IMAGES_DIR, LABELS_DIR, processor)
    train_size = int(0.9 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader    = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader      = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    optimizer   = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    total_steps = len(train_loader) * EPOCHS // ACCUM_STEPS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    scaler      = GradScaler()

    writer = SummaryWriter(LOG_DIR)
    tb     = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', LOG_DIR])
    url    = tb.launch()
    print(f"TensorBoard listening on {url}")

    # Initialize resume variables
    best_cer     = float('inf')
    best_wer     = float('inf')
    best_score   = float('-inf')
    start_epoch  = 1
    global_step  = 0

    if os.path.isfile(CHECKPOINT):
        print(f"Loading checkpoint {CHECKPOINT}")
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        best_cer    = ckpt.get('best_cer', best_cer)
        best_wer    = ckpt.get('best_wer', best_wer)
        best_score  = ckpt.get('best_score', best_score)
        start_epoch = ckpt.get('epoch', 0) + 1
        global_step = ckpt.get('global_step', 0)
        print(f"Resuming from epoch {start_epoch}, best CER {best_cer*100:.2f}%, best WER {best_wer*100:.2f}%")

    for epoch in range(start_epoch, EPOCHS + 1):
        # ------- TRAIN -------
        model.train()
        optimizer.zero_grad()
        for step, (pixel_values, labels, _, _) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        ):
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss    = outputs.loss / ACCUM_STEPS

            scaler.scale(loss).backward()
            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                writer.add_scalar('Loss/train', loss.item() * ACCUM_STEPS, global_step)
                if global_step % IMAGE_LOG_FREQ == 0:
                    imgs = pixel_values[:4]
                    grid = make_grid(imgs, nrow=2, normalize=True, value_range=(0,1))
                    writer.add_image('Train/InputGrid', grid, global_step)

        # ------- VALIDATION -------
        model.eval()
        preds, gts, keys = [], [], []
        for pixel_values, _, texts, batch_keys in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            with torch.no_grad():
                gen_ids = model.generate(pixel_values, max_length=MAX_LENGTH)
                decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
            preds.extend(decoded)
            gts.extend(texts)
            keys.extend(batch_keys)

        # CER & WER
        epoch_cer = compute_cer(preds, gts)
        valid     = [(g, p) for g, p in zip(gts, preds) if g.strip()]
        if valid:
            all_refs = " ".join([g for g, _ in valid])
            all_hyps = " ".join([p for _, p in valid])
            epoch_wer = wer(all_refs, all_hyps)
            if not math.isfinite(epoch_wer):
                epoch_wer = sum(wer(g, p) for g, p in valid) / len(valid)
        else:
            epoch_wer = float('nan')

        # Field & Document stats
        (correct_fields, wrong_fields, total_fields,
         correct_docs,   wrong_docs,   total_docs) = compute_field_and_document_stats(preds, gts, keys)

        # Percents
        cer_pct       = epoch_cer * 100
        wer_pct       = epoch_wer * 100
        field_acc_pct = correct_fields / total_fields * 100 if total_fields else 0.0
        doc_acc_pct   = correct_docs   / total_docs   * 100 if total_docs   else 0.0

        # Composite score
        final_score = (
            0.35 * (100 - wer_pct) +
            0.35 * (100 - cer_pct) +
            0.15 * field_acc_pct +
            0.15 * doc_acc_pct
        )

        # Print metrics and counts
        print(f"\nValidation — Epoch {epoch}")
        print(f"  CER: {cer_pct:.2f}%, WER: {wer_pct:.2f}%")
        print(f"  Fields: {correct_fields}/{total_fields} correct, {wrong_fields} wrong ({field_acc_pct:.2f}%)")
        print(f"  Documents: {correct_docs}/{total_docs} correct, {wrong_docs} wrong ({doc_acc_pct:.2f}%)")
        print(f"  Final composite score: {final_score:.2f}\n")

        # TensorBoard logs
        writer.add_scalar('CER/val_pct', cer_pct, epoch)
        writer.add_scalar('WER/val_pct', wer_pct, epoch)
        writer.add_scalar('Accuracy/field_pct', field_acc_pct, epoch)
        writer.add_scalar('Accuracy/document_pct', doc_acc_pct, epoch)
        writer.add_scalar('Metrics/FinalScore', final_score, epoch)

        # Show a few mismatches
        mismatch_idxs = [i for i, (p, g) in enumerate(zip(preds, gts)) if p.strip() != g.strip()]
        if mismatch_idxs:
            sampled = random.sample(mismatch_idxs, min(10, len(mismatch_idxs)))
            print(f"  Showing {len(sampled)} random mismatches:")
            for rank, idx in enumerate(sampled, 1):
                print(f"    Example {rank} [key={keys[idx]}]: GT→{gts[idx]} | Pred→{preds[idx]}")
        else:
            print("  No mismatches to show!")

        # Save latest checkpoint
        ckpt = {
            'epoch':           epoch,
            'global_step':     global_step,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state':    scaler.state_dict(),
            'best_cer':        best_cer,
            'best_wer':        best_wer,
            'best_score':      best_score
        }
        torch.save(ckpt, CHECKPOINT)
        print(f"  Checkpoint saved to {CHECKPOINT}")

        # Update best model if composite score improved
        if final_score > best_score:
            best_score = final_score
            best_cer   = epoch_cer
            best_wer   = epoch_wer
            torch.save(ckpt, BEST_CHECKPOINT)
            print(f"  → New BEST model! CER={best_cer*100:.2f}%, WER={best_wer*100:.2f}% saved to {BEST_CHECKPOINT}")

    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    main()
