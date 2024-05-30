from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train/weights/best.pt")

# Validate the model
if __name__ == '__main__':
    metrics = model.val()  # no arguments needed, dataset and settings remembered

