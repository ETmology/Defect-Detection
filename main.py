from ultralytics import YOLO

# Create a new YOLO model from scratch
# model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Run inference on the source
results = model(source=1, show=True, conf=0.4, save=True)
