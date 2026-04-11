from ultralytics import YOLO
from pathlib import Path

# Load model 
MODEL_PATH = Path("runs/classify/outputs/runs/sterile_vision_v12/weights/best.pt")
model = YOLO(str(MODEL_PATH))

# Test images
test_images = {
    "fceps.jpg"      : "clean",
    "fceps_2.png"    : "clean",
    "fceps_cont.jpg" : "contaminated",
}

print(f"\n{'='*50}")
print(f"  REAL WORLD IMAGE TEST")
print(f"{'='*50}")

for filename, expected in test_images.items():
    img_path = Path("data/test_images") / filename
    results  = model.predict(str(img_path), verbose=False)

    for r in results:
        pred = r.names[r.probs.top1]
        conf = float(r.probs.top1conf)
        correct = pred.lower() == expected.lower()

        print(f"\n  Image:    {filename}")
        print(f"  Expected: {expected.upper()}")
        print(f"  Predicted:{pred.upper()}")
        print(f"  Confidence: {conf:.2%}")
        print(f"  Result:   {'PASS' if correct else 'FAIL'}")

print(f"\n{'='*50}")