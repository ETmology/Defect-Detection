import os
import comet_ml
from ultralytics import YOLO

# 设置离线模式，结束后上传结果
os.environ["COMET_MODE"] = "offline"

# 设置项目名称
projectName = 'Defect-Detection-exp'


# 定义模型初次调优参数
def tune_model():
    # 初始化Comet项目
    comet_ml.init(project_name=projectName)

    # 加载模型
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')

    # 模型调优
    model.tune(
        data='NEU-DET.yaml',  # 数据集的yaml文件
        project=projectName,
        device='0',  # 使用的GPU设备编号
    )


if __name__ == '__main__':
    tune_model()
