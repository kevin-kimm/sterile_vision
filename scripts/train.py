from ultralytics import YOLO

# Config 
DATA    = "data/contamination_dataset"
MODEL   = "yolov8n-cls.pt"   # nano classification model
EPOCHS  = 50
IMGSZ   = 224
BATCH   = 32
NAME    = "sterile_vision_v1"

# Train
model = YOLO(MODEL)

results = model.train(
    data    = DATA,
    epochs  = EPOCHS,
    imgsz   = IMGSZ,
    batch   = BATCH,
    name    = NAME,
    project = "outputs/runs",
    device  = "mps",        # Apple M4 GPU
    patience= 10,           # early stop
    save    = True,
    exist_ok = True,  # prevents creating new version every run
)

print("\nTraining complete")
print(f"Results saved to outputs/runs/{NAME}")