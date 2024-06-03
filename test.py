from ultralytics import YOLO


# 定义测试参数
def test_model():
    # 加载模型
    model = YOLO('Defect-Detection-exp/train/weights/best.pt')

    # 测试模型
    model.val(
        data='NEU-DET.yaml',  # 数据集的yaml文件
        split='test',  # 分割出测试集
        device='0',  # 使用的GPU设备编号

        save_json=True,  # 是否保存JSON格式的结果

        batch=16,  # 批量大小
        epochs=200,  # 训练轮数
        imgsz=640,  # 输入图像尺寸

        conf=0.25
    )


if __name__ == '__main__':
    test_model()
