from ultralytics import YOLO

# 1. 加载模型
model = YOLO(r"D:\EagleGrab\runs\detect\obb_train\train_run\weights\best.pt")

# 2. 导出为 ONNX
model.export(
    format='onnx',
    imgsz=(640, 640),  # v8.1 必须写成 tuple
    opset=12,          # v8.1 推荐 opset=12，兼容性最好
)
