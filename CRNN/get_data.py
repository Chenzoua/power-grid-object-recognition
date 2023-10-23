import os
import random
from shutil import copyfile

# 设置输入文件夹和输出文件夹的路径
input_folder = 'data_own/image'  # 替换为包含图像的文件夹路径
output_folder = 'data_own/images'  # 替换为输出训练集和测试集的文件夹路径

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取文件夹中所有图片文件的列表
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]

# 随机打乱图片文件列表
random.shuffle(image_files)

# 计算分割点，划分为70%的训练集和30%的测试集
split_point = int(0.7 * len(image_files))

# 分割训练集和测试集
train_files = image_files[:split_point]
test_files = image_files[split_point:]

# 创建训练集和测试集文件夹
train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 复制训练集图片到训练集文件夹，并生成训练集标签文件
with open(os.path.join(output_folder, 'train.txt'), 'w') as train_labels_file:
    for image_file in train_files:
        src_path = os.path.join(input_folder, image_file)
        dest_path = os.path.join(train_folder, image_file)
        copyfile(src_path, dest_path)
        train_labels_file.write(f'{image_file}\n')

# 复制测试集图片到测试集文件夹，并生成测试集标签文件
with open(os.path.join(output_folder, 'test.txt'), 'w') as test_labels_file:
    for image_file in test_files:
        src_path = os.path.join(input_folder, image_file)
        dest_path = os.path.join(test_folder, image_file)
        copyfile(src_path, dest_path)
        test_labels_file.write(f'{image_file}\n')

print('数据集制作完成。')
