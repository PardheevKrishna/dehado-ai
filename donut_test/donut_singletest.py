import os
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# ================================
# CONFIGURATION
# ================================
MODEL_CHECKPOINT = "./donut_best_checkpoint.pt"   # your fine-tuned Donut checkpoint
MAX_LENGTH       = 256
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# LOAD Donut Processor + Model
# ================================
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
processor.image_processor.size = {"height": 384, "width": 384}

model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
# unwrap your fine-tuned checkpoint
torch_checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
state_dict = (
    torch_checkpoint.get("model_state", torch_checkpoint.get("model_state_dict", torch_checkpoint))
)
model.load_state_dict(state_dict)
model.to(DEVICE).eval()

# if GPU is available, use half precision for speed
use_half = False
if DEVICE.type == "cuda":
    model.half()
    use_half = True

# ================================
# INFERENCE FUNCTION FOR WHOLE IMAGE
# ================================
def infer_single(image_path: str) -> str:
    """
    Runs Donut on the entire image and returns the generated text string.
    """
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    pixel_values = inputs.pixel_values
    if use_half:
        pixel_values = pixel_values.half()

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=MAX_LENGTH)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return text

# ================================
# EXAMPLE USAGE
# ================================
if __name__ == "__main__":
    # Change this to your test image path
    test_image = "./test_img3.png"
    result_text = infer_single(test_image)
    print(f"Predicted structured output for {os.path.basename(test_image)}:")
    print(result_text)
