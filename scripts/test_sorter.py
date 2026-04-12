from ultralytics import YOLO
from pathlib import Path

# Load sorter model 
MODEL_PATH = Path("runs/classify/outputs/runs/sterile_sorter_v1/weights/best.pt")
model = YOLO(str(MODEL_PATH))

# Test images 
test_images = {
    "episiomity_01.jpeg" : "Episiotomy Scissors",
    "forceps_01.jpg"     : "Hemostat",
    "hemostat_01.png"    : "Hemostat",
    "scalpel_01.jpeg"    : "Scalpel",
    "scissors_01.png"    : "Mayo",
}

print(f"\n{'='*55}")
print(f"  TOOL SORTER TEST")
print(f"{'='*55}")

passed = 0
for filename, expected in test_images.items():
    img_path = Path("data/test_images/clean") / filename
    r = model.predict(str(img_path), verbose=False)[0]
    pred = r.names[r.probs.top1]
    conf = float(r.probs.top1conf)
    correct = pred.lower() == expected.lower()
    if correct:
        passed += 1

    print(f"\n  Image:    {filename}")
    print(f"  Expected: {expected}")
    print(f"  Predicted:{pred}")
    print(f"  Confidence: {conf:.2%}")
    print(f"  Result:   {'PASS' if correct else 'FAIL'}")

print(f"\n{'='*55}")
print(f"  TOTAL: {passed}/5 correct ({passed/5:.2%})")
print(f"{'='*55}")