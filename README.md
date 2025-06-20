# Dehado AI OCR Pipeline

This repository contains utilities for cropping handwritten text regions from documents, training a Donut-based OCR model, and evaluating the results. The workflow is:

1. **Crop** regions of interest from full-page images.
2. **Optionally clean or edit** the cropped dataset.
3. **Train** a Donut model on the cropped fields.
4. **Evaluate** or run inference on new images.

```
.
├── donut_cleaning/        # data preparation tools
├── donut_train/           # training script
├── donut_test/            # evaluation & inference scripts
└── temp.md                # notes on data format
```

## Requirements

Install the Python dependencies from `requirements.txt`. For CUDA 11.8 users,
first install PyTorch with the official wheels:

```bash
pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining packages:

```bash
pip install -r requirements.txt
```


=======

=======
Install Python packages (exact versions can be pinned in `requirements.txt` if provided):

```bash
pip install torch transformers tqdm pillow opencv-python ultralytics tensorboard jiwer

```

## Dataset Preparation

Place the raw images and JSON labels in the following structure:

```
DEHADO-AI_TRAINING_DATASET_COMPLETE/
├── IMAGES_1500/
└── LABELS_1500/
```

Run the cropping script to generate individual field images and their text labels:

```bash
python donut_cleaning/crop.py
```

Crops will be written to `dehado_cropped_dataset/images` and `dehado_cropped_dataset/labels`, each entry named after the source image and field name.

Other helpers in `donut_cleaning` include `remove_extra_spaces.py` (normalizes whitespace) and `editimages.py` (interactive crop editor). Some utilities like `find_files_with_keyword.py` launch Windows applications; they may require adjustments on other platforms.

## Training

Train a Donut model on the cropped dataset:

```bash
python donut_train/train.py
```

Key configuration options are defined at the top of `train.py` (dataset path, model name, batch size, epochs, etc.). Training logs and metrics (CER, WER, field accuracy) are saved to TensorBoard in `runs/`.

## Testing & Inference

│   ├── handwritten_model.pth   # Donut weights
│   └── best_yolov8s.pt         # YOLOv8 weights
├── src/
│   └── inference.py            # entry point for testing
├── data/                       # put images here
├── outputs/                    # JSON predictions
└── requirements.txt            # package list
The `src/inference.py` script expects your model weights in `submission/model/` and will write JSON outputs to `submission/outputs/` by default.

```bash
python donut_test/donut_singletest.py --image path/to/image.png
```

For the full detection + recognition pipeline that outputs JSON predictions:

```bash
python donut_test/donut_test_submission.py
```

A few evaluation scripts use `os.startfile`, which works only on Windows. On other systems, you may need to comment out those lines or replace them with platform-appropriate alternatives.

## Notes

- Adjust model hyperparameters (e.g., `MAX_LENGTH`, `BATCH_SIZE`) as needed.
- Ensure the dataset directories match those defined in the scripts.
- The YOLO-based detection step in `donut_test_submission.py` requires suitable model weights from `ultralytics`.


=======

## Submission Guidelines

When preparing your entry for the DeHaDo-AI Challenge, organize the files as shown below and zip them using the name `TeamName_DeHaDo-AI_Challenge.zip`:

```
submission/
├── model/
│   ├── handwritten_model.pth   # Donut weights
│   └── best_yolov8s.pt         # YOLOv8 weights
├── src/
│   └── inference.py            # entry point for testing
├── data/                       # put images here
├── outputs/                    # JSON predictions
└── requirements.txt            # package list
```

The `src/inference.py` script expects your model weights in `submission/model/` and will write JSON outputs to `submission/outputs/` by default.

Make sure to provide a short technical report summarizing your approach and an evaluation metrics report with character error rate (CER), word error rate (WER), field accuracy, document accuracy, and efficiency numbers.
=======
│   └── handwritten_model.pth       # your fine-tuned weights (required)
│   └── handwritten_model.onnx      # optional export
├── inference.py                    # entry point for testing
├── data/
│   └── sample_input/               # sample images for verification (optional)
└── requirements.txt                # package list
```

The `inference.py` script expects your model weights in `submission/model/` and will write JSON outputs to `submission/outputs/` by default.

Make sure to provide a short technical report summarizing your approach and an evaluation metrics report with character error rate (CER), word error rate (WER), field accuracy, document accuracy, and efficiency numbers.
=======

