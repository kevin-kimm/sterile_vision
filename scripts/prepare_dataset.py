import os
import shutil
import random
from pathlib import Path

# paths 
SOURCE = Path.home() / "Desktop/ia_final_proj/SURGICAL TOOLS"
DEST   = Path("data/images")

# split ratios 
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10

random.seed(42)

# copy & split
for class_folder in sorted(SOURCE.iterdir()):
    if not class_folder.is_dir():
        continue

    images = list(class_folder.glob("*.jpg")) + \
             list(class_folder.glob("*.png")) + \
             list(class_folder.glob("*.jpeg"))
    random.shuffle(images)

    n       = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:]
    }

    for split, files in splits.items():
        dest_dir = DEST / split / class_folder.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, dest_dir / f.name)
        print(f"  {split:6} | {class_folder.name:30} | {len(files):4} images")

print("\ndone")