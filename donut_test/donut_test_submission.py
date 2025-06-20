import os
import json
import time
from glob import glob
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import DonutProcessor, VisionEncoderDecoderModel, GenerationConfig
from tqdm import tqdm
import psutil

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

IMAGES_DIR       = "./test_images"           # directory with input images
OUTPUT_DIR       = "./outputs"              # where to save JSONs and summary
YOLO_MODEL_PATH  = "best_yolov8s.pt"         # your trained YOLOv8 model
DONUT_MODEL_NAME = "naver-clova-ix/donut-base"
DONUT_CHECKPOINT = "../donut_train/donut_best_checkpoint.pt"
MAX_LENGTH       = 256
NUM_BEAMS        = 5
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------
def load_models():
    # YOLOv8 detector
    yolo = YOLO(YOLO_MODEL_PATH)

    # Donut processor
    processor = DonutProcessor.from_pretrained(DONUT_MODEL_NAME)
    processor.image_processor.size = {"height": 512, "width": 512}

    # Donut model
    model = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL_NAME)
    ckpt = torch.load(DONUT_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(DEVICE).eval()

    # Configure generation separately
    gen_config = GenerationConfig(
        max_length=MAX_LENGTH,
        num_beams=NUM_BEAMS,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.eos_token),
        decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.cls_token)
    )
    model.generation_config = gen_config

    return yolo, processor, model


def detect_regions(yolo, image):
    results = yolo(image, verbose=False)
    return results[0].boxes.xyxy.cpu().numpy()


def recognize_text_in_region(processor, model, crop):
    pixel_values = processor(images=crop, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        gen_ids = model.generate(
            pixel_values,
            generation_config=model.generation_config,
            return_dict_in_generate=False
        )
    pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return pred.strip('"')

# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_paths = glob(os.path.join(IMAGES_DIR, "*.*"))

    yolo, processor, model = load_models()

    times, mems = [], []
    for img_path in tqdm(image_paths, desc="Processing images"):
        start = time.time()
        proc = psutil.Process(os.getpid())

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        image = Image.open(img_path).convert("RGB")
        boxes = detect_regions(yolo, image)

        entries = []
        for xmin, ymin, xmax, ymax in boxes:
            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
            crop = image.crop(bbox)
            text = recognize_text_in_region(processor, model, crop)
            entries.append({"text": text, "bbox": bbox})

        json_path = os.path.join(OUTPUT_DIR, f"{img_name}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(entries, jf, ensure_ascii=False, indent=4)

        elapsed = time.time() - start
        mem_mb = proc.memory_info().rss / (1024 * 1024)
        times.append(elapsed)
        mems.append(mem_mb)

    Tavg = sum(times) / len(times) if times else 0
    Mavg = sum(mems) / len(mems) if mems else 0
    efficiency_score = 1 / (Tavg * Mavg) if Tavg > 0 and Mavg > 0 else 0

    summary = {
        "Tavg_s": Tavg,
        "Mavg_MB": Mavg,
        "efficiency_score": efficiency_score,
        "efficiency_units": "1/(s*MB)",
        "num_beams": NUM_BEAMS
    }
    summary_path = os.path.join(OUTPUT_DIR, "efficiency_summary.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=4)

if __name__ == "__main__":
    main()