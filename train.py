import os
import comet_ml
from ultralytics import YOLO
import yaml

# os.environ["COMET_MODE"] = "offline"  # 设置离线模式，结束后上传结果
os.environ["COMET_API_KEY"] = "YnYyHOYRurdu1KdGoAetHJxl4"

# 设置项目名称
projectName = 'Defect-Detection-exp'


# 定义训练参数
def train_model():
    # 初始化Comet项目
    comet_ml.init(project_name=projectName)

    # 加载模型
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # 重新构建
    # model = YOLO('yolov8n.pt')  # 继续构建

    # 加载超参数
    with open('runs/tune3/tune/best_hyperparameters.yaml', 'r', encoding='utf-8') as f:
        best_hyperparameters = yaml.safe_load(f)

    # 训练模型
    model.train(
        data='NEU-DET.yaml',  # 数据集的yaml文件
        project=projectName,
        device='0',  # 使用的GPU设备编号
        save_json=True,  # 是否保存JSON格式的结果

        batch=16,  # 批量大小
        epochs=100,  # 训练轮数
        imgsz=640,  # 输入图像尺寸

        # 进行调优后的超参数
        lr0=best_hyperparameters['lr0'],  # 初始学习率
        lrf=best_hyperparameters['lrf'],  # 最终学习率
        momentum=best_hyperparameters['momentum'],  # SGD 或 Adam 优化器的动量因子
        weight_decay=best_hyperparameters['weight_decay'],  # L2 正则化项
        warmup_epochs=best_hyperparameters['warmup_epochs'],  # 学习率热身阶段的时期数
        warmup_momentum=best_hyperparameters['warmup_momentum'],  # 热身阶段的初始动量
        box=best_hyperparameters['box'],  # 目标框损失在总损失中的权重
        cls=best_hyperparameters['cls'],  # 分类损失在总损失中的权重
        dfl=best_hyperparameters['dfl'],  # 分布焦点损失在总损失中的权重
        hsv_h=best_hyperparameters['hsv_h'],  # 色相调整
        hsv_s=best_hyperparameters['hsv_s'],  # 饱和度调整
        hsv_v=best_hyperparameters['hsv_v'],  # 亮度调整
        degrees=best_hyperparameters['degrees'],  # 随机旋转角度范围
        translate=best_hyperparameters['translate'],  # 随机平移比例
        scale=best_hyperparameters['scale'],  # 图像缩放因子
        shear=best_hyperparameters['shear'],  # 剪切角度范围
        perspective=best_hyperparameters['perspective'],  # 透视变换参数
        flipud=best_hyperparameters['flipud'],  # 上下翻转概率
        fliplr=best_hyperparameters['fliplr'],  # 左右翻转概率
        bgr=best_hyperparameters['bgr'],  # RGB 到 BGR 通道翻转概率
        mosaic=best_hyperparameters['mosaic'],  # 马赛克数据增强概率
        mixup=best_hyperparameters['mixup'],  # MixUp 数据增强概率
        copy_paste=best_hyperparameters['copy_paste']  # 复制粘贴数据增强概率

    )


if __name__ == '__main__':
    train_model()
