import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd

# 定义XML文件所在目录
xml_base_dir = 'NEU-DET/annotations/'

# 初始化一个字典来存储类别及其出现次数
class_counts = {
    'test': defaultdict(int),
    'train': defaultdict(int),
    'val': defaultdict(int)
}

# 遍历test、train和val文件夹
for folder_name in ['test', 'train', 'val']:
    # 定义当前文件夹的路径
    xml_dir = os.path.join(xml_base_dir, folder_name)

    # 遍历当前文件夹中的每个文件
    for root, dirs, files in os.walk(xml_dir):
        for file in files:
            if file.endswith('.xml'):
                # 解析XML文件
                tree = ET.parse(os.path.join(root, file))
                xml_root = tree.getroot()  # 将根节点命名为xml_root

                # 遍历XML文件中的每个<object>标签
                for obj in xml_root.findall('object'):
                    # 获取类别名称
                    class_name = obj.find('name').text

                    # 将类别计数加1
                    class_counts[folder_name][class_name] += 1

# 将结果写入txt文件
with open('dataset_stats_xml.txt', 'w') as f:
    for folder_name, class_count in class_counts.items():
        f.write(f'{folder_name.capitalize()} set:\n')
        for class_name, count in class_count.items():
            f.write(f'{class_name}: {count}\n')
        f.write('\n')
