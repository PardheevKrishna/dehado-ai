# Dehado-AI

An end-to-end Python project for automated donut detection, classification, and analysis.

## Project Overview

This repository contains utilities and scripts to preprocess images, train and evaluate a donut detection model, and generate submission outputs. Key components include:

- **donut_cleaning/**: Image preprocessing scripts to clean and prepare dataset images.
- **donut_train/**: Training scripts and checkpoints for model development using PyTorch and YOLOv8.
- **donut_test/**: Inference scripts and test harness to evaluate the model on sample images and produce JSON results.
- **submission/**: Submission template, code, and helper files for creating final outputs.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optional but recommended)

### Installation

1. Clone the repository:
   ```powershell
   git clone https://github.com/PardheevKrishna/dehado-ai.git
   cd dehado-ai
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Repository Structure

```
README.md
requirements.txt

Donut Cleaning:
  donut_cleaning/
    ├── crop.py                # Crop images to ROI
    ├── editimages.py          # Adjust brightness, contrast
    ├── find_files_with_keyword.py
    ├── move_files.py          # Organize files into folders
    └── remove_extra_spaces.py # Clean file names and paths

Training Pipeline:
  donut_train/
    ├── train.py               # Main training script
    ├── donut_checkpoint.pt    # Latest model checkpoint
    └── donut_best_checkpoint.pt

Testing & Inference:
  donut_test/
    ├── donut_test.py          # Batch inference and JSON output
    ├── donut_singletest.py    # Single image inference
    ├── donut_test_submission.py # Generate formatted submission
    └── outputs/               # Example output JSONs

Submission Folder:
  submission/
    ├── data/
    ├── model/
    ├── output_json_files/
    └── src/                  # Submission-specific scripts
```

## Usage

### 1. Data Cleaning

```powershell
python donut_cleaning/crop.py 
python donut_cleaning/remove_extra_spaces.py 
# Other cleaning scripts as needed
```

### 2. Model Training

```powershell
python donut_train/train.py
```

### 3. Inference & Testing

```powershell
python donut_test/donut_test.py 
```

