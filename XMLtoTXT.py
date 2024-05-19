import os
import xml.etree.ElementTree as ET


# 将XML格式的标注文件转换为YOLO格式
def convert_xml_to_yolo(xml_file, classes, output_folder):
    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 获取图像文件名和尺寸
    image_filename = root.find('filename').text
    image_width = int(root.find('size').find('width').text)
    image_height = int(root.find('size').find('height').text)

    # 初始化存储YOLO格式标签的列表
    yolo_labels = []

    # 遍历XML文件中的每个对象
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in classes:  # 如果对象类别不在指定的类别中，则跳过
            continue

        # 获取对象类别的ID和边界框信息
        class_id = classes.index(class_name)
        bbox = obj.find('bndbox')

        # 计算YOLO格式的标签信息
        x_center = (int(bbox.find('xmin').text) + int(bbox.find('xmax').text)) / 2 / image_width
        y_center = (int(bbox.find('ymin').text) + int(bbox.find('ymax').text)) / 2 / image_height
        w = (int(bbox.find('xmax').text) - int(bbox.find('xmin').text)) / image_width
        h = (int(bbox.find('ymax').text) - int(bbox.find('ymin').text)) / image_height

        # 将YOLO格式的标签信息添加到列表中
        yolo_labels.append(f"{class_id} {x_center} {y_center} {w} {h}")

    # 拼接输出文件路径
    output_path = os.path.join(output_folder, os.path.splitext(image_filename)[0] + ".txt")

    # 将YOLO格式的标签信息写入到文件中
    with open(output_path, 'w') as out_file:
        for line in yolo_labels:
            out_file.write(line + '\n')


# 定义类别列表
classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

train_xml_dir = 'NEU-DET/annotations/train'  # 训练集XML文件所在目录路径
train_output_dir = 'NEU_DET_data/train'  # 训练集YOLO格式标签文件输出目录

test_xml_dir = 'NEU-DET/annotations/test'  # 测试集XML文件所在目录路径
test_output_dir = 'NEU_DET_data/test'  # 测试集YOLO格式标签文件输出目录

# 如果输出目录不存在，则创建
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# 转换训练集格式
for xml_file in os.listdir(train_xml_dir):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(train_xml_dir, xml_file)
        convert_xml_to_yolo(xml_path, classes, train_output_dir)

# 转换测试集格式
for xml_file in os.listdir(test_xml_dir):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(test_xml_dir, xml_file)
        convert_xml_to_yolo(xml_path, classes, test_output_dir)
