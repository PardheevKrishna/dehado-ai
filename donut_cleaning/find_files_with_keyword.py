import subprocess
from pathlib import Path

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
# Set the search word (case-insensitive) to look for in filenames (stem)
SEARCH_WORD      = "date".lower()
# Set a word to exclude (case-insensitive) even if it contains SEARCH_WORD
EXCLUDE_WORD     = "candidatename".lower()

LABELS_DIR  = Path("./dehado_cropped_dataset/labels")
IMAGES_DIR  = LABELS_DIR.parent / "images"
IMAGE_EXT   = ".jpg"
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # Gather all label .txt files
    txt_files = list(LABELS_DIR.rglob("*.txt"))
    # Filter based on SUBSTRING matching, excluding any with EXCLUDE_WORD
    matching = [txt for txt in txt_files
                if SEARCH_WORD in txt.stem.lower()
                and EXCLUDE_WORD not in txt.stem.lower()]

    total = len(matching)
    print(f"Found {total} files with '{SEARCH_WORD}', excluding '{EXCLUDE_WORD}'.")

    # Iterate over each matching file with index
    for idx, txt in enumerate(matching, start=1):
        print(f"Processing {idx}/{total}: {txt.name}")
        img_path = IMAGES_DIR / (txt.stem + IMAGE_EXT)

        # Open the label file in Notepad (non-blocking)
        if txt.exists():
            print(f"Opening label file: {txt}")
            subprocess.Popen(["notepad", str(txt)])
        else:
            print(f"Label file not found: {txt}")

        # Open the image in MS Paint and wait until closed
        if img_path.exists():
            print(f"Opening image in MS Paint: {img_path}")
            subprocess.call(["mspaint", str(img_path)])
            print(f"Image closed: {img_path}")
        else:
            print(f"Image file not found: {img_path}")

    print("All done.")

if __name__ == "__main__":
    main()
