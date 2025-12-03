from ultralytics import YOLO
import os

DATA_YAML = r"d:/EagleGrab/dataset2/data2.yaml"
MODEL_PATH = r"d:/EagleGrab/yolov8n-obb.pt"
SAVE_DIR = r"d:/EagleGrab/dataset2/train_obb"

os.makedirs(SAVE_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

model.train(
    data=DATA_YAML,
    epochs=30,
    imgsz=640,
    batch=8,
    project=SAVE_DIR,
    name="train_run",
    exist_ok=True,
    device='cpu',
    task='obb'
)

print("✅ 训练完成，结果保存在:", os.path.join(SAVE_DIR, "train_run"))

