import os
import shutil
from tqdm import tqdm

def copy_all_files(src_dir: str, dest_dir: str) -> None:
    """
    Copy all files from src_dir to dest_dir with a progress bar.

    :param src_dir: Path to the source directory.
    :param dest_dir: Path to the destination directory.
    """
    # Ensure source exists
    if not os.path.isdir(src_dir):
        raise ValueError(f"Source directory does not exist: {src_dir}")

    # Create destination directory if needed
    os.makedirs(dest_dir, exist_ok=True)

    # Gather all file names in src_dir (skip folders)
    all_entries = os.listdir(src_dir)
    files = [f for f in all_entries if os.path.isfile(os.path.join(src_dir, f))]

    # Copy each file with a progress bar
    for filename in tqdm(files, desc="Copying files", unit="file"):
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        try:
            shutil.copy2(src_path, dest_path)
        except Exception as e:
            # Log the error, but keep going
            tqdm.write(f"⚠️ Failed to copy {filename}: {e}")

if __name__ == "__main__":
    # Example usage: adjust these paths to your actual folders
    source_folder = "DEHADO-AI_TRAINING_DATASET/IMAGES_750"
    destination_folder = "DEHADO-AI_TRAINING_DATASET_COMPLETE\IMAGES_1500"
    copy_all_files(source_folder, destination_folder)
