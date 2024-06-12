import os
import comet_ml
import yaml
from ultralytics import YOLO

# os.environ["COMET_MODE"] = "offline"  # 设置离线模式，结束后上传结果
os.environ["COMET_API_KEY"] = "YnYyHOYRurdu1KdGoAetHJxl4"  # 实时上传结果

# 设置项目名称
projectName = 'Defect-Detection-exp'


# 定义训练参数
def train_model():
    # 初始化Comet项目
    comet_ml.init(project_name=projectName)

    # 加载模型
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # 重新构建

    # 加载超参数
    with open('hyp.yaml', 'r', encoding='utf-8') as f:
        hyp = yaml.safe_load(f)

    # 训练模型
    model.train(
        data='NEU-DET.yaml',  # 数据集的yaml文件
        project=projectName,
        device='0',  # 使用的GPU设备编号
        save_json=True,  # 是否保存JSON格式的结果

        batch=16,  # 批量大小
        epochs=100,  # 训练轮数
        imgsz=320,  # 输入图像尺寸

        # 进行调优后的超参数
        lr0=hyp['lr0'],  # 初始学习率
        lrf=hyp['lrf'],  # 最终学习率
        momentum=hyp['momentum'],  # SGD 或 Adam 优化器的动量因子
        weight_decay=hyp['weight_decay'],  # L2 正则化项
        warmup_epochs=hyp['warmup_epochs'],  # 学习率热身阶段的时期数
        warmup_momentum=hyp['warmup_momentum'],  # 热身阶段的初始动量
        box=hyp['box'],  # 目标框损失在总损失中的权重
        cls=hyp['cls'],  # 分类损失在总损失中的权重
        dfl=hyp['dfl'],  # 分布焦点损失在总损失中的权重
        hsv_h=hyp['hsv_h'],  # 色相调整
        hsv_s=hyp['hsv_s'],  # 饱和度调整
        hsv_v=hyp['hsv_v'],  # 亮度调整
        degrees=hyp['degrees'],  # 随机旋转角度范围
        translate=hyp['translate'],  # 随机平移比例
        scale=hyp['scale'],  # 图像缩放因子
        shear=hyp['shear'],  # 剪切角度范围
        perspective=hyp['perspective'],  # 透视变换参数
        flipud=hyp['flipud'],  # 上下翻转概率
        fliplr=hyp['fliplr'],  # 左右翻转概率
        bgr=hyp['bgr'],  # RGB 到 BGR 通道翻转概率
        mosaic=hyp['mosaic'],  # 马赛克数据增强概率
        mixup=hyp['mixup'],  # MixUp 数据增强概率
        copy_paste=hyp['copy_paste'],  # 复制粘贴数据增强概率
    )


if __name__ == '__main__':
    train_model()
