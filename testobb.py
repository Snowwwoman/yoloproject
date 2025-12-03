from ultralytics import YOLO
import os

# 路径配置
MODEL_PATH = r"D:\EagleGrab\dataset2\train_obb\train_run\weights\best.pt"  # 训练好的模型
IMAGES_DIR = r"d:/EagleGrab/dataset2/images/test"  # 测试集图片目录
SAVE_DIR = r"d:/EagleGrab/dataset2/test_obb"  # 推理结果保存目录

os.makedirs(SAVE_DIR, exist_ok=True)

# 加载模型
model = YOLO(MODEL_PATH)

# 对验证集图片批量推理
results = model.predict(
    source=IMAGES_DIR,
    save=True,
    save_txt=True,
    project=SAVE_DIR,
    name="test_run",
    exist_ok=True,
    device='cpu',
    imgsz=640,
    conf=0.50,
    iou=0.5,
    task='obb'
)

print("✅ 推理完成，结果保存在:", os.path.join(SAVE_DIR, "test_run"))
