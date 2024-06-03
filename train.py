import os
import comet_ml
from ultralytics import YOLO

# 设置Comet API Key
os.environ['COMET_API_KEY'] = 'YnYyHOYRurdu1KdGoAetHJxl4'

# 设置项目名称
projectName = 'Defect-Detection'


def train_model():
    # 初始化Comet项目
    comet_ml.init(project_name=projectName)

    # 加载模型
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')

    # 训练模型
    results = model.train(
        data='NEU-DET.yaml',  # 数据集的yaml文件
        project=projectName,
        batch=32,
        save_period=1,
        save_json=True,
        epochs=3,  # 训练轮数
        imgsz=200,  # 输入图像尺寸
        device='0'  # 使用的GPU设备编号
    )


if __name__ == '__main__':
    train_model()

exit()
