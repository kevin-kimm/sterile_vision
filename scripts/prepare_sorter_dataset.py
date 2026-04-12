import shutil
import random
from pathlib import Path

# Paths 
SOURCE = Path("data/images")
DEST   = Path("data/sorted_images")

# 7 classes — no Overlapping 
CLASSES = [
    "Episiotomy Scissors",
    "Forceps",
    "Hemostat",
    "Mayo",
    "Scalpel",
    "Stitch Scissors",
    "Syringe",
]

# Split ratios 
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10

random.seed(42)

# Clear old dataset 
if DEST.exists():
    shutil.rmtree(DEST)
    print("Cleared old sorted dataset\n")

# Build dataset
print("Building sorter dataset...\n")

for cls in CLASSES:
    # Gather all images across all splits
    all_images = []
    for split in ["train", "val", "test"]:
        folder = SOURCE / split / cls
        if folder.exists():
            all_images += list(folder.glob("*.jpg"))  + \
                          list(folder.glob("*.png"))  + \
                          list(folder.glob("*.jpeg"))

    random.shuffle(all_images)

    n       = len(all_images)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": all_images[:n_train],
        "val":   all_images[n_train:n_train + n_val],
        "test":  all_images[n_train + n_val:]
    }

    for split, files in splits.items():
        dest_dir = DEST / split / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, dest_dir / f.name)
        print(f"  {split:6} | {cls:25} | {len(files):4} images")

print("\nDataset ready at data/sorted_images/")