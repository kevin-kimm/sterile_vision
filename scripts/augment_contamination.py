import cv2
import numpy as np
import random
from pathlib import Path
import shutil

# Paths 
SOURCE      = Path("data/images")
DEST        = Path("data/contamination_dataset")
OVERLAPPING = Path("data/images/train/Overlapping")

SPLITS = ["train", "val", "test"]

random.seed(42)
np.random.seed(42)

def blood_color():
    """Realistic blood color palette."""
    choice = random.random()
    if choice < 0.5:
        # Fresh blood — red
        return (random.randint(0, 20),
                random.randint(0, 30),
                random.randint(140, 190))
    elif choice < 0.85:
        # Dried blood — dark red/brown
        return (random.randint(0, 15),
                random.randint(15, 45),
                random.randint(90, 140))
    else:
        # Old dried — dark brown
        return (random.randint(10, 30),
                random.randint(20, 50),
                random.randint(60, 100))


def add_blood_stain(image):
    """Add realistic blood stain focused on tool area."""
    img = image.copy()
    h, w = img.shape[:2]

    cx_min, cx_max = int(w * 0.2), int(w * 0.8)
    cy_min, cy_max = int(h * 0.2), int(h * 0.8)

    overlay = img.copy()

    # Small irregular blobs 
    num_blobs = random.randint(3, 8)
    for _ in range(num_blobs):
        cx = random.randint(cx_min, cx_max)
        cy = random.randint(cy_min, cy_max)

        rx = random.randint(2, max(3, w // 25))
        ry = random.randint(2, max(3, h // 25))
        color = blood_color()

        num_points = random.randint(5, 9)
        angles = np.linspace(0, 2 * np.pi, num_points)
        points = []
        for angle in angles:
            r = random.uniform(0.4, 1.0)
            px = int(cx + rx * r * np.cos(angle))
            py = int(cy + ry * r * np.sin(angle))
            points.append([px, py])
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(overlay, [points], color)

        # Splatter drops
        for _ in range(random.randint(2, 8)):
            angle = random.uniform(0, 2 * np.pi)
            dist  = random.randint(3, max(4, rx * 3))
            dx = int(cx + dist * np.cos(angle))
            dy = int(cy + dist * np.sin(angle))
            dr = random.randint(1, 3)
            dx = max(0, min(w - 1, dx))
            dy = max(0, min(h - 1, dy))
            cv2.circle(overlay, (dx, dy), dr, color, -1)

    alpha = random.uniform(0.5, 0.8)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Occasional smear
    if random.random() < 0.25:
        x1 = random.randint(cx_min, cx_max)
        y1 = random.randint(cy_min, cy_max)
        x2 = x1 + random.randint(-40, 40)
        y2 = y1 + random.randint(-10, 10)
        cv2.line(img, (x1, y1), (x2, y2), blood_color(),
                 random.randint(1, 4))

    return img


def process_split(split):
    clean_count        = 0
    synth_cont_count   = 0
    real_cont_count    = 0

    split_source = SOURCE / split

    if not split_source.exists():
        return

    for class_folder in sorted(split_source.iterdir()):
        if not class_folder.is_dir():
            continue

        # Skip Overlapping — handled separately 
        if class_folder.name == "Overlapping":
            continue

        images = list(class_folder.glob("*.jpg"))  + \
                 list(class_folder.glob("*.png"))  + \
                 list(class_folder.glob("*.jpeg"))

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Save clean version
            clean_dest = DEST / split / "clean" / class_folder.name
            clean_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, clean_dest / img_path.name)
            clean_count += 1

            # Save synthetically contaminated version
            contaminated = add_blood_stain(img)
            cont_dest = DEST / split / "contaminated" / "synthetic" / class_folder.name
            cont_dest.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cont_dest / img_path.name), contaminated)
            synth_cont_count += 1

    # Copy real Overlapping images as contaminated 
    overlap_source = SOURCE / split / "Overlapping"
    if overlap_source.exists():
        overlap_images = list(overlap_source.glob("*.jpg"))  + \
                         list(overlap_source.glob("*.png"))  + \
                         list(overlap_source.glob("*.jpeg"))

        for img_path in overlap_images:
            real_dest = DEST / split / "contaminated" / "real"
            real_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, real_dest / img_path.name)
            real_cont_count += 1

    total_cont = synth_cont_count + real_cont_count
    print(f"\n  {split.upper()}")
    print(f"    clean:              {clean_count:5} images")
    print(f"    contaminated (real):{real_cont_count:5} images")
    print(f"    contaminated (syn): {synth_cont_count:5} images")
    print(f"    contaminated total: {total_cont:5} images")


# Run
if DEST.exists():
    shutil.rmtree(DEST)
    print("🗑️  Cleared old dataset\n")

print("Building contamination dataset...\n")
for split in SPLITS:
    process_split(split)

print("\nDone. Dataset saved to data/contamination_dataset/")