from ultralytics import YOLO


# 定义训练参数
def tune_model():
    # 加载模型
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')

    # 模型调优
    model.tune(
        data='NEU-DET.yaml',  # 数据集的yaml文件
        device='0',  # 使用的GPU设备编号
        epochs=30,
        iterations=300,
        optimizer="AdamW",
        plots=False,
        save=False,
        val=False
    )


if __name__ == '__main__':
    tune_model()
