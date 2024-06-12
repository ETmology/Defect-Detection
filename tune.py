from ultralytics import YOLO
import yaml


# 定义训练参数
def tune_model():
    # 加载模型
    model = YOLO('Defect-Detection-exp/train_2.1.1/weights/best.pt')

    # 加载超参数
    with open('runs/tune2/tune/best_hyperparameters.yaml', 'r', encoding='utf-8') as f:
        best_hyperparameters = yaml.safe_load(f)

    # 模型调优
    model.tune(
        data='NEU-DET.yaml',  # 数据集的yaml文件
        project="runs/tune3",  # 修改项目名称
        device='0',  # 使用的GPU设备编号
        epochs=30,
        iterations=100,
        optimizer="AdamW",
        plots=False,
        save=False,
        val=False,

        # 进行调优后的超参数
        lr0=best_hyperparameters['lr0'],
        lrf=best_hyperparameters['lrf'],
        momentum=best_hyperparameters['momentum'],
        weight_decay=best_hyperparameters['weight_decay'],
        warmup_epochs=best_hyperparameters['warmup_epochs'],
        warmup_momentum=best_hyperparameters['warmup_momentum'],
        box=best_hyperparameters['box'],
        cls=best_hyperparameters['cls'],
        dfl=best_hyperparameters['dfl'],
        hsv_h=best_hyperparameters['hsv_h'],
        hsv_s=best_hyperparameters['hsv_s'],
        hsv_v=best_hyperparameters['hsv_v'],
        degrees=best_hyperparameters['degrees'],
        translate=best_hyperparameters['translate'],
        scale=best_hyperparameters['scale'],
        shear=best_hyperparameters['shear'],
        perspective=best_hyperparameters['perspective'],
        flipud=best_hyperparameters['flipud'],
        fliplr=best_hyperparameters['fliplr'],
        bgr=best_hyperparameters['bgr'],
        mosaic=best_hyperparameters['mosaic'],
        mixup=best_hyperparameters['mixup'],
        copy_paste=best_hyperparameters['copy_paste']
    )


if __name__ == '__main__':
    tune_model()
