from ultralytics import YOLO

# Load best model
model = YOLO("runs/classify/outputs/runs/sterile_vision_v12/weights/best.pt")

# Test images, same filename, different folder
clean_img = "data/contamination_dataset/test/clean/1012_png.rf.f1e69a08a3f80b10381fa2aa7516a025.jpg"
cont_img  = "data/contamination_dataset/test/contaminated/1012_png.rf.f1e69a08a3f80b10381fa2aa7516a025.jpg"

for label, path in [("CLEAN", clean_img), ("CONTAMINATED", cont_img)]:
    results = model.predict(path, verbose=False)
    for r in results:
        probs = r.probs
        pred  = r.names[probs.top1]
        conf  = probs.top1conf
        print(f"\nActual:     {label}")
        print(f"Predicted:  {pred.upper()}")
        print(f"Confidence: {conf:.2%}")
        print(f"Correct:    {'PASS' if pred.lower() == label.lower() else 'FAIL'}")