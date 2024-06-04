# 环境配置

1. Pytorch `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
2. Yolov8  `pip install ultralytics`
3. (optional) Comet[^1] `pip install comet_ml` 

> [!note]
>
> Comet设置成了离线模式，每次train结束后，先在终端设置API `set COMET_API_KEY=YnYyHOYRurdu1KdGoAetHJxl4`，再按照提示上传本地`.cometml-runs`中的文件即可。

# 项目文件说明

## NEU-DET

| 缺陷类型        | 描述       | 外观特征                                                     |
| :-------------- | :--------- | :----------------------------------------------------------- |
| crazing         | 开裂       | 一种在轧件表面呈现的不连续裂纹，从中心点向外呈闪电状发散。   |
| inclusion       | 夹杂       | 板带钢表面的薄层折叠，缺陷带呈灰白色，其大小和形状各异，不规则分布于板带钢表面。 |
| patches         | 斑块       | 带钢表面出现片状或大面积的斑迹，有时在某个角度上有向外辐射的迹象。 |
| pitted_surface  | 点蚀表面   | 带钢表面局部或连续出现粗糙面，严重时呈现桔皮状。在上下表面均可能出现，且在带钢长度方向上的密度不均。 |
| rolled-in_scale | 轧制氧化皮 | 通常以小斑点、鱼鳞状、条状或块状不规则分布于带钢上、下表面的全部或局部，常伴有粗糙的麻点状表面。 |
| scratches       | 划痕       | 轧件表面的机械损伤，长度、宽度和深度各异，主要沿轧制方向或垂直于轧制方向出现。 |

# 相关文档

## 模型参数

> [!note]
>
> 基于COCO数据集的检测（COCO），参阅[检测文档](https://docs.ultralytics.com/tasks/detect/)，了解在[COCO](https://docs.ultralytics.com/datasets/detect/coco/)上训练的这些模型的用法示例，其中包括 80 个预训练类别。

| Model                                                        | size (pixels) | mAPval 50-95 | Speed CPU ONNX (ms) | Speed A100 TensorRT (ms) | params (M) | FLOPs (B) |
| ------------------------------------------------------------ | ------------- | ------------ | ------------------- | ------------------------ | ---------- | --------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt) | 640           | 37.3         | 80.4                | 0.99                     | 3.2        | 8.7       |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt) | 640           | 44.9         | 128.4               | 1.20                     | 11.2       | 28.6      |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt) | 640           | 50.2         | 234.7               | 1.83                     | 25.9       | 78.9      |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt) | 640           | 52.9         | 375.2               | 2.39                     | 43.7       | 165.2     |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt) | 640           | 53.9         | 479.1               | 3.53                     | 68.2       | 257.8     |

- **mAPval** 值是在[COCO val2017](https://cocodataset.org/)数据集上进行单一模型单一尺度测试的结果。
  可通过 `yolo val detect data=coco.yaml device=0` 来重现。
- **速度** 平均值是在[COCO val](https://aws.amazon.com/ec2/instance-types/p4/)图像上使用[Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)实例进行测试的结果。
  可通过 `yolo val detect data=coco.yaml batch=1 device=0|cpu` 来重现。

## 性能指标

### 目标检测性能指标

| 指标                | 描述                                                         |
| ------------------- | ------------------------------------------------------------ |
| 交并比（IoU）       | IoU是衡量预测边界框与真实边界框重叠程度的指标。它在评估目标定位准确性方面起着基础性作用。 |
| 平均精度（AP）      | AP计算精度-召回率曲线下的面积，提供一个单一值来概括模型的精度和召回率表现。 |
| 平均精度均值（mAP） | mAP通过计算多个目标类别的平均AP值来扩展AP的概念。在多类目标检测场景中，这对于提供模型性能的综合评估非常有用。 |
| 精度（Precision）   | Precision（精度）衡量了所有正例预测中正例的比例，评估了模型避免假正例的能力。 |
| 召回率（Recall）    | Recall（召回率）计算了所有实际正例中正例的比例，衡量了模型检测某一类别的所有实例的能力。 |
| F1分数              | F1分数是精度和召回率的调和平均数，提供了一个平衡考虑假阳性和假阴性的模型性能评估。 |

> [!tip]
>
> 选择正确的指标进行评估通常取决于具体的应用：
>
> - **mAP**：适用于对模型性能进行广泛评估。
> - **IoU**：当精确的对象位置至关重要时是必不可少的。
> - **精度**：当最小化错误检测是首要任务时非常重要。
> - **召回率**：当重要性检测每个对象实例时至关重要。
> - **F1分数**：在需要精度和召回率之间平衡时非常有用。

对于实时应用程序，像FPS（每秒帧数）和延迟这样的速度指标至关重要，以确保及时获得结果。

### 解释输出

#### 按类别的指标（Class-wise Metrics）

这个部分提供了对模型在每个特定类别上的性能指标的细分信息。这对于理解模型在数据集中每个特定类别上的表现有很大帮助，特别是在包含多种目标类别的数据集中。对于数据集中的每个类别，提供以下信息：

- **类别（Class）**：表示对象类别的名称，例如“person”、“car”或“dog”。
- **图像数（Images）**：这个指标告诉你在验证集中包含该对象类别的图像数量。
- **实例数（Instances）**：这提供了该类别在验证集中出现的总次数。
- **Box(P, R, mAP50, mAP50-95)**[^2]：这些指标提供了模型在检测对象方面的性能信息：
    - **精度（Precision）**：检测到的对象的准确性，表示有多少检测是正确的。
    - **召回率（Recall）**：模型在图像中识别所有对象实例的能力。
    - **mAP50**：在交并比（IoU）阈值为0.50时计算的平均精度。这是考虑“容易”检测的模型准确性的度量。
    - **mAP50-95**：在不同IoU阈值范围（从0.50到0.95）计算的平均平均精度的平均值。它提供了模型在不同检测难度级别上的综合表现。

#### 速度指标（Speed Metrics）

略。

#### 视觉输出

除了生成数字指标之外，`model.val()`函数还产生了可视化输出，这些输出可以更直观地理解模型的性能。

1. **F1分数曲线**（F1_curve.png）：该曲线表示不同阈值下的F1分数。解读这条曲线可以提供有关模型在不同阈值下假阳性和假阴性之间平衡的洞察。

2. **精度-召回率曲线**（PR_curve.png）：对于任何分类问题来说，这是一个重要的可视化工具，该曲线展示了在不同阈值下精度和召回率之间的权衡。在处理不平衡类别时，这变得尤为重要。

3. **精度曲线**（P_curve.png）：在不同阈值下显示精度值的图形表示。这条曲线有助于理解随着阈值变化，精度如何变化。

4. **召回率曲线**（R_curve.png）：相应地，该图表说明了召回率值在不同阈值下的变化。

5. **混淆矩阵**（confusion_matrix.png）：混淆矩阵提供了结果的详细视图，展示了每个类别的真正例、真负例、假正例和假负例的计数。

6. **归一化混淆矩阵**（confusion_matrix_normalized.png）：这种可视化是混淆矩阵的归一化版本。它表示数据的比例而不是原始计数。这种格式使得比较类别间的性能更加简单。

7. **验证批次标签**（val_batchX_labels.jpg）：这些图像描述了来自验证数据集的不同批次的地面真实标签。它们清晰地展示了对象是什么以及它们在数据集中的位置。

8. **验证批次预测**（val_batchX_pred.jpg）：与标签图像形成对比，这些可视化显示了YOLOv8模型对相应批次的预测。通过将这些与标签图像进行比较，您可以轻松评估模型在视觉上检测和分类对象的表现如何。

## 训练过程中的损失函数

在YOLOv8中，常见的损失函数包括分类损失（cls_loss）、边界框损失（box_loss）和回归损失（dfl_loss），这些损失函数在训练过程中被同时优化，以帮助模型学习正确地预测目标类别和边界框位置。

1. **分类损失（cls_loss）**：用于衡量模型在预测目标类别时的误差。分类损失通常使用交叉熵损失函数来计算，用于衡量模型预测的类别与实际类别之间的差异。
2. **边界框损失（box_loss）**：用于衡量模型在预测目标边界框位置时的误差。边界框损失通常使用平方误差损失函数来计算，用于衡量模型预测的边界框与实际边界框之间的差异。
3. **回归损失（dfl_loss）**：用于衡量模型在预测目标边界框细节时的误差。回归损失通常使用平方误差损失函数来计算，用于衡量模型预测的边界框细节（如中心点、宽度和高度）与实际边界框细节之间的差异。

# 待办

- [x] 阅读Ultralytics官方文档
  - [x] 计算机视觉项目的步骤：[Steps of a Computer Vision Project - Ultralytics YOLO Docs](https://docs.ultralytics.com/guides/steps-of-a-cv-project/)
  - [x] 性能指标：[YOLO Performance Metrics - Ultralytics YOLO Docs](https://docs.ultralytics.com/guides/yolo-performance-metrics/)
  - [x] 超参数调整：[Hyperparameter Tuning - Ultralytics YOLO Docs](https://docs.ultralytics.com/guides/hyperparameter-tuning/)
- [ ] 分析`train`结果
- [ ] 调整n型的超参数
- [ ] 试验s、m、l、x型号的模型

# 参考

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/modes/)
- [Ultralytics Github](https://github.com/ultralytics/ultralytics)

---

[^1]: 实验记录平台
[^2]: 精度（Precision）、召回率（Recall）和mAP指标（mAP50和mAP50-95），其中“Box”表示目标检测中的边界框
