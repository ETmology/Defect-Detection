from ultralytics import YOLO
import comet_ml
import yaml
import os

os.environ["COMET_API_KEY"] = "YnYyHOYRurdu1KdGoAetHJxl4"  # 实时上传结果

# 设置项目名称
projectName = 'Defect-Detection-exp'


# 定义训练参数
def tune_model():
    # 初始化Comet项目
    comet_ml.init(project_name=projectName)

    # 加载模型
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # 重新构建

    # 加载超参数
    with open('runs/tune4/tune/best_hyperparameters.yaml', 'r', encoding='utf-8') as f:
        hyp = yaml.safe_load(f)

    # 模型调优
    model.tune(
        data='NEU-DET.yaml',  # 数据集的yaml文件
        project=projectName,
        device='0',  # 使用的GPU设备编号
        iterations=80,
        optimizer="AdamW",
        # plots=False,
        # save=False,
        # val=False,

        save_json=True,  # 是否保存JSON格式的结果

        batch=16,  # 批量大小
        epochs=100,  # 训练轮数
        imgsz=256,  # 输入图像尺寸

        # 进行调优后的超参数
        lr0=hyp['lr0'],
        lrf=hyp['lrf'],
        momentum=hyp['momentum'],
        weight_decay=hyp['weight_decay'],
        warmup_epochs=hyp['warmup_epochs'],
        warmup_momentum=hyp['warmup_momentum'],
        box=hyp['box'],
        cls=hyp['cls'],
        dfl=hyp['dfl'],
        hsv_h=hyp['hsv_h'],
        hsv_s=hyp['hsv_s'],
        hsv_v=hyp['hsv_v'],
        degrees=hyp['degrees'],
        translate=hyp['translate'],
        scale=hyp['scale'],
        shear=hyp['shear'],
        perspective=hyp['perspective'],
        flipud=hyp['flipud'],
        fliplr=hyp['fliplr'],
        bgr=hyp['bgr'],
        mosaic=hyp['mosaic'],
        mixup=hyp['mixup'],
        copy_paste=hyp['copy_paste']
    )


if __name__ == '__main__':
    tune_model()
