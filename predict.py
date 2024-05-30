from ultralytics import YOLO
import os

# 这个不太用得到，用validation即可

# 加载模型
model = YOLO("runs/detect/train/weights/best.pt")  # 加载已训练好的 YOLO 模型

# 测试图片文件夹的路径
test_dir = "test"

# 结果保存文件夹的路径
output_dir = "result"

# 如果结果保存文件夹不存在，则创建
os.makedirs(output_dir, exist_ok=True)

# 获取测试文件夹中所有图片文件的列表
image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f)) and
               f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 对每个图片文件进行推理
for image_file in image_files:
    # 对当前图片进行推理
    results = model(image_file)

    # 将带有边界框的结果图片保存到输出文件夹中
    filename = os.path.join(output_dir, "result_" + os.path.basename(image_file))
    results[0].save(filename)  # 保存第一个结果图片
