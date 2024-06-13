import os
import shutil

# 源数据集路径
source_dir = r'D:\Mine\Coding\pythonProjects\Defect-Detection\datasets\random_split_20'
# 目标数据集路径
target_dir = r'D:\Mine\Coding\pythonProjects\Defect-Detection\datasets\rolled-in_scale_samples'

# 创建目标数据集目录结构
os.makedirs(os.path.join(target_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'test'), exist_ok=True)

# 指定要提取的类别
hard_class = 4


def extract_hard_samples(split):
    split_dir = os.path.join(source_dir, split)
    target_split_dir = os.path.join(target_dir, split)

    if not os.path.exists(split_dir):
        print(f"目录 {split_dir} 不存在，请检查路径。")
        return

    for file in os.listdir(split_dir):
        if file.endswith('.txt'):
            label_path = os.path.join(split_dir, file)
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # 保留类别为0的行
            lines = [line.strip() for line in lines if line.startswith(str(hard_class))]

            if len(lines) > 0:
                # 复制标注文件
                with open(os.path.join(target_split_dir, file), 'w') as f:
                    f.write('\n'.join(lines))

                # 复制对应的图像文件
                image_file = file.replace('.txt', '.jpg')
                image_path = os.path.join(split_dir, image_file)
                if os.path.exists(image_path):
                    shutil.copy(image_path, target_split_dir)


# 提取训练集、验证集和测试集中类别为0的样本
extract_hard_samples('train')
extract_hard_samples('val')
extract_hard_samples('test')

print('提取完成')
