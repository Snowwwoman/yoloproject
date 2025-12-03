from ultralytics import YOLO

model_path = r"d:/EagleGrab/dataset2/train_obb/train_run/weights/best.pt"
print("正在加载模型：", model_path)

model = YOLO(model_path)

print("模型 labels =", model.names)
