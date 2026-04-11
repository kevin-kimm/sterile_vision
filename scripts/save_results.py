import json
import csv
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

# Load model
MODEL_PATH = Path("runs/classify/outputs/runs/sterile_vision_v12/weights/best.pt")
TEST_DIR   = Path("data/contamination_dataset/test")
OUTPUT_DIR = Path("outputs/logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL_PATH))

correct = 0
total   = 0
wrong   = []
all_results = []

for true_label in ["clean", "contaminated"]:
    img_dir = TEST_DIR / true_label
    images  = list(img_dir.glob("*.jpg")) + \
              list(img_dir.glob("*.png"))

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

            all_results.append({
                "image":      img_path.name,
                "true_label": true_label,
                "predicted":  pred,
                "confidence": round(conf, 4),
                "correct":    is_correct,
            })

# Summary
accuracy  = correct / total
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

summary = {
    "timestamp":  timestamp,
    "model":      str(MODEL_PATH),
    "total":      total,
    "correct":    correct,
    "wrong":      len(wrong),
    "accuracy":   round(accuracy, 4),
    "misclassified": wrong
}

# Save JSON summary 
json_path = OUTPUT_DIR / "evaluation_summary.json"
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"JSON summary saved to {json_path}")

# Save full CSV report
csv_path = OUTPUT_DIR / "evaluation_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image","true_label","predicted","confidence","correct"])
    writer.writeheader()
    writer.writerows(all_results)
print(f"CSV report saved to {csv_path}")

# Print summary 
print(f"\n{'='*40}")
print(f"  EVALUATION RESULTS")
print(f"{'='*40}")
print(f"  Timestamp:     {timestamp}")
print(f"  Total images:  {total}")
print(f"  Correct:       {correct}")
print(f"  Wrong:         {len(wrong)}")
print(f"  Accuracy:      {accuracy:.2%}")
print(f"{'='*40}")