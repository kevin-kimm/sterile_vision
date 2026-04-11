from ultralytics import YOLO
from pathlib import Path

# Load model
MODEL_PATH = Path("runs/classify/outputs/runs/sterile_vision_v12/weights/best.pt")
TEST_DIR   = Path("data/contamination_dataset/test")

print(f"Model exists: {MODEL_PATH.exists()}")
print(f"Test dir exists: {TEST_DIR.exists()}")

model = YOLO(str(MODEL_PATH))

correct = 0
total   = 0
wrong   = []

for true_label in ["clean", "contaminated"]:
    img_dir = TEST_DIR / true_label
    images  = list(img_dir.glob("*.jpg")) + \
              list(img_dir.glob("*.png"))

    print(f"\n{true_label}: {len(images)} images found")

    for img_path in images:
        results = model.predict(str(img_path), verbose=False)
        for r in results:
            pred       = r.names[r.probs.top1]
            conf       = float(r.probs.top1conf)
            is_correct = pred.lower() == true_label.lower()

            if is_correct:
                correct += 1
            else:
                wrong.append({
                    "image":      img_path.name,
                    "true_label": true_label,
                    "predicted":  pred,
                    "confidence": round(conf, 4),
                })
            total += 1

# Print summary 
print(f"\n{'='*40}")
print(f"  EVALUATION RESULTS")
print(f"{'='*40}")
print(f"  Total images:  {total}")
print(f"  Correct:       {correct}")
print(f"  Wrong:         {len(wrong)}")
print(f"  Accuracy:      {correct/total:.2%}")
print(f"{'='*40}")

if wrong:
    print(f"\n  Misclassified images:")
    for w in wrong:
        print(f"  - {w['image']}")
        print(f"    True: {w['true_label']} | Pred: {w['predicted']} | Conf: {w['confidence']:.2%}")
else:
    print("\n  No misclassifications")