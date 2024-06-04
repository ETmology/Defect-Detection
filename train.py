import os
import comet_ml
from ultralytics import YOLO

# 设置离线模式，结束后上传结果
os.environ["COMET_MODE"] = "offline"

# 设置项目名称
projectName = 'Defect-Detection-exp'


# 定义训练参数
def train_model():
    # 初始化Comet项目
    comet_ml.init(project_name=projectName)

    # 加载模型
    model = YOLO('yolov8s.yaml').load('pre_models/yolov8s.pt')

    # 训练模型
    model.train(
        data='NEU-DET.yaml',  # 数据集的yaml文件
        project=projectName,
        device='0',  # 使用的GPU设备编号
        save_json=True,  # 是否保存JSON格式的结果

        batch=16,  # 批量大小
        epochs=200,  # 训练轮数
        imgsz=640,  # 输入图像尺寸

        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率
        momentum=0.937,  # 动量
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3,  # 初始warmup轮数
        warmup_momentum=0.8,  # 初始warmup动量
        warmup_bias_lr=0.1,  # 初始warmup偏置学习率
    )


if __name__ == '__main__':
    train_model()
