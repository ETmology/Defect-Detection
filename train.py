from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.yaml').load('yolov8n.pt')

# 训练模型
if __name__ == '__main__':
    # 指定训练数据集的路径和其他参数
    data = 'NEU-DET.yaml'  # 数据集的yaml文件
    epochs = 200  # 训练轮数
    imgsz = 640  # 输入图像尺寸
    device = '0'  # 使用的GPU设备编号

    # 开始训练
    model.train(data=data, epochs=epochs, imgsz=imgsz, device=device)