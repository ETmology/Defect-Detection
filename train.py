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
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')

    # 训练模型
    model.train(
        data='NEU-DET.yaml',  # 数据集的yaml文件
        project=projectName,
        device='0',  # 使用的GPU设备编号
        save_json=True,  # 是否保存JSON格式的结果

        batch=16,  # 批量大小
        epochs=200,  # 训练轮数
        imgsz=640,  # 输入图像尺寸

        # 进行调优后的超参数
        lr0=0.00651,
        lrf=0.00896,
        momentum=0.88183,
        weight_decay=0.00051,
        warmup_epochs=3.41933,
        warmup_momentum=0.71674,
        box=6.3139,
        cls=0.41714,
        dfl=1.66416,
        hsv_h=0.02157,
        hsv_s=0.49884,
        hsv_v=0.35053,
        degrees=0.0,
        translate=0.08529,
        scale=0.46097,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.42133,
        bgr=0.0,
        mosaic=0.97803,
        mixup=0.0,
        copy_paste=0.0
    )


if __name__ == '__main__':
    train_model()
