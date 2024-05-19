from ultralytics import YOLO

# 加载模型
# model = YOLO('yolov8s.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8s.yaml').load('yolov8s.pt')  # build from YAML and transfer weights

# 训练模型
if __name__ == '__main__':
    model.train(data='data.yaml', epochs=200, imgsz=640, device='0')  # device：指定0号GPU执行
