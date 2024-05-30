import os
import cv2
import xml.etree.ElementTree as ET


def draw_bounding_box(image_folder, annotation_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_folder, filename)
            xml_path = os.path.join(annotation_folder, filename[:-4] + '.xml')
            output_path = os.path.join(output_folder, filename)

            # 读取图片
            image = cv2.imread(image_path)

            # 解析XML文件
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 提取对象信息并绘制边界框
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                class_name = obj.find('name').text  # 提取对象类别信息
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                # 在图像上绘制边界框和类别信息
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image, class_name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # 保存绘制好边界框的图片
            cv2.imwrite(output_path, image)


# 绘制边界框并保存图片
draw_bounding_box('NEU-DET/images/train', 'NEU-DET/annotations/train', 'draw_xml_boxes/train')
draw_bounding_box('NEU-DET/images/val', 'NEU-DET/annotations/val', 'draw_xml_boxes/val')
draw_bounding_box('NEU-DET/images/test', 'NEU-DET/annotations/test', 'draw_xml_boxes/test')
