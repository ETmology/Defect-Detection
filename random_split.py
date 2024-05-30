import os
import shutil
from sklearn.model_selection import train_test_split

# 数据集根目录
dataset_root = 'NEU-DET'

# 训练集和验证集目录
train_ann_dir = os.path.join(dataset_root, 'annotations', 'train')
train_img_dir = os.path.join(dataset_root, 'images', 'train')
train_label_dir = os.path.join(dataset_root, 'labels', 'train')

val_ann_dir = os.path.join(dataset_root, 'annotations', 'val')
val_img_dir = os.path.join(dataset_root, 'images', 'val')
val_label_dir = os.path.join(dataset_root, 'labels', 'val')
os.makedirs(val_ann_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 获取训练集中的所有文件
train_ann_files = os.listdir(train_ann_dir)
train_img_files = os.listdir(train_img_dir)
train_label_files = os.listdir(train_label_dir)

# 随机划分训练集和验证集，验证集占比20%，随机种子42保证每次划分结果一致
train_ann_files, val_ann_files = train_test_split(train_ann_files, test_size=300, random_state=42)
train_img_files, val_img_files = train_test_split(train_img_files, test_size=300, random_state=42)
train_label_files, val_label_files = train_test_split(train_label_files, test_size=300, random_state=42)

# 将验证集文件移动到验证集目录
for file in val_ann_files:
    src = os.path.join(train_ann_dir, file)
    dst = os.path.join(val_ann_dir, file)
    shutil.move(src, dst)

for file in val_img_files:
    src = os.path.join(train_img_dir, file)
    dst = os.path.join(val_img_dir, file)
    shutil.move(src, dst)

for file in val_label_files:
    src = os.path.join(train_label_dir, file)
    dst = os.path.join(val_label_dir, file)
    shutil.move(src, dst)

print("Validation set created successfully.")
