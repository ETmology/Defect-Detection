# Defect-Detection
> [!note]
>
> Course Assignment

### Env Config

1. Pytorch `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
2. YoloV8  `pip install ultralytics`
3. Comet `pip install comet_ml`

### Comet Config

1. 设置API `set COMET_API_KEY=YnYyHOYRurdu1KdGoAetHJxl4`
2. 结束后按照提示上传本地`.cometml-runs`中的文件即可

### NEU-DET Classes

1. **Crazing** 裂纹
2. **Inclusion** 夹杂
3. **Patches** 斑块
4. **Pitted Surface** 坑洼面
5. **Rolled-in Scale** 轧制鳞片
6. **Scratches** 划痕

### Ref

- [YOLOv8实现缺陷目标检测（附代码、数据集、教学视频）](https://zhuanlan.zhihu.com/p/666040746)
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/modes/)