import shutil
from pathlib import Path

DEST = Path("data/contamination_dataset")

for split in ["train", "val", "test"]:
    cont_dir = DEST / split / "contaminated"
    
    # Move all images from real/ and synthetic/ up to contaminated/
    for subfolder in ["real", "synthetic"]:
        sub_path = cont_dir / subfolder
        if not sub_path.exists():
            continue
        
        # Walk all images in subfolder (including nested class folders)
        for img_path in sub_path.rglob("*.jpg"):
            shutil.move(str(img_path), str(cont_dir / img_path.name))
        for img_path in sub_path.rglob("*.png"):
            shutil.move(str(img_path), str(cont_dir / img_path.name))
        for img_path in sub_path.rglob("*.jpeg"):
            shutil.move(str(img_path), str(cont_dir / img_path.name))
        
        # Remove now-empty subfolder
        shutil.rmtree(sub_path)
        print(f"Flattened {split}/contaminated/{subfolder}/")

    # Also flatten clean/ subfolders
    clean_dir = DEST / split / "clean"
    for img_path in clean_dir.rglob("*.jpg"):
        shutil.move(str(img_path), str(clean_dir / img_path.name))
    for img_path in clean_dir.rglob("*.png"):
        shutil.move(str(img_path), str(clean_dir / img_path.name))
    
    # Remove empty class subfolders in clean/
    for subfolder in clean_dir.iterdir():
        if subfolder.is_dir():
            shutil.rmtree(subfolder)
    print(f"Flattened {split}/clean/")

print("\nDataset flattened ready for YOLO classification")