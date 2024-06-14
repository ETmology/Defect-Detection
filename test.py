from ultralytics import YOLO

project_name = "Defect-Detection-exp/train_3.2.3"


# 定义测试参数
def test_model():
    # 加载模型
    model = YOLO(f'{project_name}/weights/best.pt')

    # 测试模型
    model.val(
        data='NEU-DET.yaml',  # 数据集的yaml文件
        split='test',  # 分割出测试集
        device='0',  # 使用的GPU设备编号
        project=project_name,
        imgsz=320,

        save_json=True,  # 是否保存JSON格式的结果
        conf=0.25  # 信心阈值
    )


if __name__ == '__main__':
    test_model()
