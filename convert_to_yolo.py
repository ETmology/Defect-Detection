import os
from xml.etree import ElementTree as ET
from PIL import Image


def convert_annotation(xml_path, image_path, label_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image = Image.open(image_path)
    image_width, image_height = image.size

    label_map = {
        'crazing': 0,
        'inclusion': 1,
        'patches': 2,
        'pitted_surface': 3,
        'rolled-in_scale': 4,
        'scratches': 5
    }

    with open(label_path, 'w') as label_file:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = label_map[class_name]
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            x_center = (xmin + xmax) / (2 * image_width)
            y_center = (ymin + ymax) / (2 * image_height)
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            label_file.write(f'{class_id} {x_center} {y_center} {width} {height}\n')


# 数据集根目录
root_dir = 'NEU-DET'
annotations_dir = os.path.join(root_dir, 'annotations')
images_dir = os.path.join(root_dir, 'images')
labels_dir = os.path.join(root_dir, 'labels')

# 转换train数据集
train_annotations_dir = os.path.join(annotations_dir, 'train')
train_images_dir = os.path.join(images_dir, 'train')
train_labels_dir = os.path.join(labels_dir, 'train')
os.makedirs(train_labels_dir, exist_ok=True)

for filename in os.listdir(train_annotations_dir):
    if filename.endswith('.xml'):
        image_name = filename[:-4] + '.jpg'
        xml_path = os.path.join(train_annotations_dir, filename)
        image_path = os.path.join(train_images_dir, image_name)
        label_path = os.path.join(train_labels_dir, filename[:-4] + '.txt')

        convert_annotation(xml_path, image_path, label_path)

# 转换test数据集
test_annotations_dir = os.path.join(annotations_dir, 'test')
test_images_dir = os.path.join(images_dir, 'test')
test_labels_dir = os.path.join(labels_dir, 'test')
os.makedirs(test_labels_dir, exist_ok=True)

for filename in os.listdir(test_annotations_dir):
    if filename.endswith('.xml'):
        image_name = filename[:-4] + '.jpg'
        xml_path = os.path.join(test_annotations_dir, filename)
        image_path = os.path.join(test_images_dir, image_name)
        label_path = os.path.join(test_labels_dir, filename[:-4] + '.txt')

        convert_annotation(xml_path, image_path, label_path)
