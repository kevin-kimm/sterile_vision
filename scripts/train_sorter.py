from ultralytics import YOLO

# Config 
DATA    = "data/sorted_images"
MODEL   = "yolov8n-cls.pt"
EPOCHS  = 50
IMGSZ   = 224
BATCH   = 32
NAME    = "sterile_sorter_v1"

# Train 
model = YOLO(MODEL)

results = model.train(
    data     = DATA,
    epochs   = EPOCHS,
    imgsz    = IMGSZ,
    batch    = BATCH,
    name     = NAME,
    project  = "outputs/runs",
    device   = "mps",
    patience = 10,
    save     = True,
    exist_ok = True,
)

print("\nTraining complete")
print(f"Results saved to outputs/runs/{NAME}")