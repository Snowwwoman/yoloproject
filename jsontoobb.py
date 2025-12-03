import os
import json

# ================= 配置 =================
annotation_dir = r"C:\Users\15900\Desktop\2025_08_29_16 展会轴件数据集\训练2"  # JSON 文件夹
image_dir = annotation_dir
output_label_dir = os.path.join(annotation_dir, "labels")
os.makedirs(output_label_dir, exist_ok=True)

# 类别映射
class_map = {
    "轴件1": 0,
    "轴件2": 1,
    "轴件23": 2
}

# ================= 遍历 JSON 文件 =================
for file_name in os.listdir(annotation_dir):
    if not file_name.endswith(".json"):
        continue

    json_path = os.path.join(annotation_dir, file_name)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data.get("imageWidth")
    img_h = data.get("imageHeight")

    txt_name = os.path.splitext(file_name)[0] + ".txt"
    txt_path = os.path.join(output_label_dir, txt_name)

    yolo_lines = []

    for shape in data.get("shapes", []):
        label = shape.get("label")
        points = shape.get("points")  # 期望为 4 个点：[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

        if label not in class_map or not points or len(points) != 4:
            continue

        cls_id = class_map[label]

        # 归一化坐标
        norm_pts = []
        for (x, y) in points:
            nx = x / img_w
            ny = y / img_h
            norm_pts.append((nx, ny))

        # YOLO OBB格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
        line = f"{cls_id} " + " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in norm_pts])
        yolo_lines.append(line)

    # 写入 YOLO OBB 标签
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))

print(f"✅ 转换完成！YOLO OBB 标签保存在: {output_label_dir}")
