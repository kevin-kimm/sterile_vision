import cv2
import numpy as np
import random
from ultralytics import YOLO
from pathlib import Path

# Setup 
MODEL_PATH  = Path("runs/classify/outputs/runs/sterile_vision_v12/weights/best.pt")
CLEAN_DIR   = Path("data/test_images/clean")
CONT_DIR    = Path("data/test_images/contaminated")
CONT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL_PATH))

random.seed(42)
np.random.seed(42)

# Same blood stain function from training 
def blood_color():
    choice = random.random()
    if choice < 0.5:
        return (random.randint(0, 20),
                random.randint(0, 30),
                random.randint(140, 190))
    elif choice < 0.85:
        return (random.randint(0, 15),
                random.randint(15, 45),
                random.randint(90, 140))
    else:
        return (random.randint(10, 30),
                random.randint(20, 50),
                random.randint(60, 100))

def add_blood_stain(image):
    img = image.copy()
    h, w = img.shape[:2]
    cx_min, cx_max = int(w * 0.2), int(w * 0.8)
    cy_min, cy_max = int(h * 0.2), int(h * 0.8)
    overlay = img.copy()

    for _ in range(random.randint(3, 8)):
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

    if random.random() < 0.25:
        x1 = random.randint(cx_min, cx_max)
        y1 = random.randint(cy_min, cy_max)
        x2 = x1 + random.randint(-40, 40)
        y2 = y1 + random.randint(-10, 10)
        cv2.line(img, (x1, y1), (x2, y2), blood_color(),
                 random.randint(1, 4))
    return img

# Generate contaminated versions 
images = list(CLEAN_DIR.glob("*.jpg"))  + \
         list(CLEAN_DIR.glob("*.jpeg")) + \
         list(CLEAN_DIR.glob("*.png"))

for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    contaminated = add_blood_stain(img)
    out_path = CONT_DIR / img_path.name
    cv2.imwrite(str(out_path), contaminated)

# Run predictions 
print(f"\n{'='*55}")
print(f"  REAL WORLD IMAGE TEST")
print(f"{'='*55}")

passed = 0
total  = 0

for img_path in sorted(images):
    name     = img_path.name
    cont_path = CONT_DIR / name

    # Test clean
    r_clean = model.predict(str(img_path), verbose=False)[0]
    pred_clean = r_clean.names[r_clean.probs.top1]
    conf_clean = float(r_clean.probs.top1conf)
    correct_clean = pred_clean.lower() == "clean"

    # Test contaminated
    r_cont = model.predict(str(cont_path), verbose=False)[0]
    pred_cont = r_cont.names[r_cont.probs.top1]
    conf_cont = float(r_cont.probs.top1conf)
    correct_cont = pred_cont.lower() == "contaminated"

    if correct_clean: passed += 1
    if correct_cont:  passed += 1
    total += 2

    print(f"\n  {name}")
    print(f"  Clean      → {pred_clean:15} {conf_clean:.2%}  {'PASS' if correct_clean else 'FAIL'}")
    print(f"  Contaminated → {pred_cont:13} {conf_cont:.2%}  {'PASS' if correct_cont else 'FAIL'}")

print(f"\n{'='*55}")
print(f"  TOTAL: {passed}/{total} correct ({passed/total:.2%})")
print(f"{'='*55}")