# import os
# import json
#
# # 设置要遍历的文件夹路径
# folder_path = "/home/zhihui/Dataset/data_poi_921/label"
#
# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     if filename.endswith(".json"):
#         # 构建完整的文件路径
#         file_path = os.path.join(folder_path, filename)
#
#         # 打开JSON文件并解析内容
#         with open(file_path, 'r') as json_file:
#             data = json.load(json_file)
#
#             # 假设标注信息存储在 'shapes' 键下
#             shapes = data.get('shapes', [])
#
#             # 遍历标注信息
#             for shape in shapes:
#                 label = shape.get("label", "")
#                 description = shape.get("description", "")
#
#                 # 检查标签是否为 "poi_val0" 或 "poi_val1"，以及 "description" 是否为空
#                 if label in ["poi_val0", "poi_val1"] and not description:
#                     # 输出文件名
#                     print(f"File '{filename}' has empty 'description' for label '{label}'.")
#

# # ----------------------数据清洗，提取标签文件
# import os
# import shutil
#
# def move_json_files(source_folder, destination_folder):
#     # 检查源文件夹是否存在
#     if not os.path.exists(source_folder):
#         print(f"源文件夹 '{source_folder}' 不存在！")
#         return
#     # 检查目标文件夹是否存在，如果不存在则创建
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)
#     # 遍历源文件夹中的所有文件
#     for filename in os.listdir(source_folder):
#         # 检查文件是否为.json文件
#         if filename.endswith(".json"):
#             # 构建源文件路径和目标文件路径
#             source_path = os.path.join(source_folder, filename)
#             destination_path = os.path.join(destination_folder, filename)
#
#             # 移动文件
#             shutil.move(source_path, destination_path)
#             print(f"移动文件 '{filename}' 成功！")
# # 调用函数进行移动操作
# source_folder = "/home/zhihui/Dataset/data_poi_921/images"
# destination_folder = "/home/zhihui/Dataset/data_poi_921/label"
# move_json_files(source_folder, destination_folder)


# # 代码描述：数据清洗第二步--- 对比数据源图（jpg）与所保存的标注的文件，打印缺失的图片序号
#
# import os
# from tqdm import tqdm
#
# xml_path = '/home/zhihui/Dataset/数据集/数据集/数据集04/lig_ol/txt'       # 标注文件路径
# image_path = '/home/zhihui/Dataset/数据集/数据集/数据集04/lig_ol/image'     # 数据原图路径
# image_lst = os.listdir(image_path)
# xml_lst = os.listdir(xml_path)
# print("image list:", len(image_lst))
# print("xml list:", len(xml_lst))
#
# # missing_index = []
# # for image in tqdm(image_lst):
# #     xml = image[:-4] + '.txt'
# #     if xml not in xml_lst:
# #         missing_index.append(xml[:-4])
# # print(len(missing_index))
# # print(missing_index)
#
# missing_index = []
# for txt in tqdm(xml_lst):
#     image = txt[:-4] + '.jpg'
#     print(image)
#     if image not in image_lst:
#         missing_index.append(image[:-4])
# print(len(missing_index))
# print(missing_index)

#
# ###################文件匹配删除
# import os
#
# # 指定包含jpg文件的文件夹路径
# jpg_folder = '/home/zhihui/Dataset/data_poi_109/images'
#
# # 指定包含json文件的文件夹路径
# json_folder = '/home/zhihui/Dataset/data_poi_109/label'
#
# # 获取jpg文件和json文件的列表
# jpg_files = [f for f in os.listdir(jpg_folder) if f.endswith('.jpg')]
# json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
#
# # 将jpg文件和json文件的文件名去掉扩展名后，用集合进行比较
# jpg_names = {os.path.splitext(f)[0] for f in jpg_files}
# json_names = {os.path.splitext(f)[0] for f in json_files}
#
# # 删除jpg文件夹中不匹配的文件
# for jpg_file in jpg_files:
#     jpg_name = os.path.splitext(jpg_file)[0]
#     if jpg_name not in json_names:
#         os.remove(os.path.join(jpg_folder, jpg_file))
#         print(f"删除不匹配的jpg文件: {jpg_file}")
#
# # 删除json文件夹中不匹配的文件
# for json_file in json_files:
#     json_name = os.path.splitext(json_file)[0]
#     if json_name not in jpg_names:
#         os.remove(os.path.join(json_folder, json_file))
#         print(f"删除不匹配的json文件: {json_file}")


#########################图片文件大小筛选
# import os
# import shutil
#
# # 源文件夹路径，包含需要筛选的图片
# source_folder = '/home/zhihui/Dataset/Dataset_Result/data_det/image'
#
# # 目标文件夹路径，用于存放筛选出的小于40KB的图片
# target_folder = '/home/zhihui/Dataset/Dataset_Result/data_det/image/temp'
#
# # 确保目标文件夹存在，如果不存在则创建它
# if not os.path.exists(target_folder):
#     os.makedirs(target_folder)
#
# # 遍历源文件夹中的文件
# for filename in os.listdir(source_folder):
#     filepath = os.path.join(source_folder, filename)
#
#     # 检查文件是否为图片（这里假定只处理.jpg文件，可以根据需要修改扩展名）
#     if filename.endswith('.jpg'):
#         # 获取文件大小（以字节为单位）
#         file_size = os.path.getsize(filepath)
#
#         # 如果文件大小小于40KB（以字节为单位）
#         if file_size < 25 * 1024:
#             # 构建目标文件的路径
#             target_filepath = os.path.join(target_folder, filename)
#
#             # 移动文件到目标文件夹
#             shutil.move(filepath, target_filepath)
#             print(f"移动文件: {filename} 到 {target_filepath}")
#
# print("筛选并移动完成。")


#
# import os
# import json
# import shutil
#
# # 源文件夹路径，包含需要遍历的JSON文件
# source_folder = '/home/zhihui/Dataset/Dataset_Result/data_det/train2'
#
# # 目标文件夹路径，用于存放包含矩形标注的JSON文件
# target_folder = '/home/zhihui/Dataset/Dataset_Result/data_det/train2/temp'
#
# # 确保目标文件夹存在，如果不存在则创建它
# if not os.path.exists(target_folder):
#     os.makedirs(target_folder)
#
# # 遍历源文件夹中的文件
# for filename in os.listdir(source_folder):
#     filepath = os.path.join(source_folder, filename)
#
#     # 检查文件是否为JSON文件（可以根据需要修改扩展名）
#     if filename.endswith('.json'):
#         # 打开JSON文件并加载内容
#         with open(filepath, 'r') as load_f:
#             load_dict = json.load(load_f)
#
#         # 在JSON文件中查找矩形标注
#         if 'shapes' in load_dict:
#             shapes = load_dict['shapes']
#
#             # 遍历形状列表
#             for shape in shapes:
#                 if 'shape_type' in shape and shape['shape_type'] == 'rectangle':
#                     # 如果存在矩形标注，将文件移动到目标文件夹
#                     target_filepath = os.path.join(target_folder, filename)
#                     shutil.move(filepath, target_filepath)
#                     print(f"移动文件: {filename} 到 {target_filepath}")
#
# print("遍历并移动完成。")

# #########修改矩形标注为多边形四点标注
# import os
# import json
#
# # 源文件夹路径，包含需要修改的JSON文件
# source_folder = '/home/zhihui/Dataset/Dataset_Result/data_det/train2/temp'
#
# # 目标文件夹路径，用于存放修改后的JSON文件
# target_folder = '/home/zhihui/Dataset/Dataset_Result/data_det/train2/temp/1'
#
# # 确保目标文件夹存在，如果不存在则创建它
# if not os.path.exists(target_folder):
#     os.makedirs(target_folder)
#
# # 遍历源文件夹中的文件
# for filename in os.listdir(source_folder):
#     filepath = os.path.join(source_folder, filename)
#
#     # 检查文件是否为JSON文件（可以根据需要修改扩展名）
#     if filename.endswith('.json'):
#         # 打开JSON文件并加载内容
#         with open(filepath, 'r') as load_f:
#             load_dict = json.load(load_f)
#
#         # 在JSON文件中查找矩形标注
#         if 'shapes' in load_dict:
#             shapes = load_dict['shapes']
#
#             # 遍历形状列表
#             for shape in shapes:
#                 if 'shape_type' in shape and shape['shape_type'] == 'rectangle':
#                     # 获取矩形的四个坐标点
#                     x1, y1 = shape['points'][0]
#                     x2, y2 = shape['points'][1]
#
#                     # 将矩形标注改为四点标注（polygon）
#                     shape['shape_type'] = 'polygon'
#                     shape['points'] = [
#                         [x1, y1],
#                         [x2, y1],
#                         [x2, y2],
#                         [x1, y2]
#                     ]
#
#                     # 将修改后的JSON保存到目标文件夹中
#                     target_filepath = os.path.join(target_folder, filename)
#                     with open(target_filepath, 'w') as save_f:
#                         json.dump(load_dict, save_f, indent=4)
#                     print(f"修改并保存文件: {filename} 到 {target_filepath}")
#
# print("遍历并修改完成。")


#################################josn文件修改，标注规范统一
# import os
# import json
#
# # 源文件夹路径，包含需要修改的JSON文件
# source_folder = '/home/zhihui/Dataset/data/annotations/train'
#
# # 目标文件夹路径，用于存放修改后的JSON文件
# target_folder = '/home/zhihui/Dataset/data/annotations/train/change'
#
# # 确保目标文件夹存在，如果不存在则创建它
# if not os.path.exists(target_folder):
#     os.makedirs(target_folder)
#
# # 遍历源文件夹中的文件
# for filename in os.listdir(source_folder):
#     filepath = os.path.join(source_folder, filename)
#
#     # 检查文件是否为JSON文件（可以根据需要修改扩展名）
#     if filename.endswith('.json'):
#         # 打开JSON文件并加载内容
#         with open(filepath, 'r') as load_f:
#             load_dict = json.load(load_f)
#
#         # 在JSON文件中查找"label"字段并修改值
#         if 'shapes' in load_dict:
#             shapes = load_dict['shapes']
#
#             for shape in shapes:
#                 if 'label' in shape:
#                     if shape['label'] == '1':
#                         shape['label'] = 'poi'
#                     elif shape['label'] == '2':
#                         shape['label'] = 'sca'
#
#             # 将修改后的JSON保存到目标文件夹中
#             target_filepath = os.path.join(target_folder, filename)
#             with open(target_filepath, 'w') as save_f:
#                 json.dump(load_dict, save_f, indent=4)
#             print(f"修改并保存文件: {filename} 到 {target_filepath}")
#
# print("遍历并修改完成。")

# import os
# import json
#
# # 源文件夹路径，包含需要修改的JSON文件
# source_folder = '/home/zhihui/Dataset/data/annotations/train1'
#
# # 目标文件夹路径，用于存放修改后的JSON文件
# target_folder = '/home/zhihui/Dataset/data/annotations/train1/change'
#
# # 确保目标文件夹存在，如果不存在则创建它
# if not os.path.exists(target_folder):
#     os.makedirs(target_folder)
#
# # 遍历源文件夹中的文件
# for filename in os.listdir(source_folder):
#     filepath = os.path.join(source_folder, filename)
#
#     # 检查文件是否为JSON文件（可以根据需要修改扩展名）
#     if filename.endswith('.json'):
#         # 打开JSON文件并加载内容
#         with open(filepath, 'r') as load_f:
#             load_dict = json.load(load_f)
#
#         # 在JSON文件中查找"label"字段并修改值
#         if 'shapes' in load_dict:
#             shapes = load_dict['shapes']
#
#             for shape in shapes:
#                 if 'label' in shape:
#                     label_value = shape['label']
#                     shape['label'] = 'poi_val'
#                     shape['description'] = label_value
#
#             # 将修改后的JSON保存到目标文件夹中
#             target_filepath = os.path.join(target_folder, filename)
#             with open(target_filepath, 'w') as save_f:
#                 json.dump(load_dict, save_f, indent=4)
#             print(f"修改并保存文件: {filename} 到 {target_filepath}")
#
# print("遍历并修改完成。")



# import os
# import shutil
#
# # 定义源文件夹和目标文件夹
# source_folder = '/home/zhihui/Dataset/data2'  # 替换为实际的源文件夹路径
# target_folder1 = '/home/zhihui/Dataset/data_2/images'  # 替换为实际的目标文件夹1路径
# target_folder2 = '/home/zhihui/Dataset/data_2/labels'  # 替换为实际的目标文件夹2路径
#
# # 遍历源文件夹及其子文件夹
# for root, dirs, files in os.walk(source_folder):
#     for file in files:
#         file_path = os.path.join(root, file)
#         # 提取.jpg文件到目标文件夹1
#         if file.endswith('.jpg'):
#             shutil.copy(file_path, os.path.join(target_folder1, file))
#         # 提取.json文件到目标文件夹2
#         elif file.endswith('.json'):
#             shutil.copy(file_path, os.path.join(target_folder2, file))
#
# print("提取完成。")
#


# import os
# import shutil
#
# # 定义源文件夹和目标文件夹
# source_folder = '/home/zhihui/Downloads/Detect-and-read-meters-2/data_detection/labels'  # 替换为实际的源文件夹路径
# target_folder = '/home/zhihui/temp'  # 目标文件夹名为temp
#
# # 创建目标文件夹，如果不存在
# if not os.path.exists(target_folder):
#     os.mkdir(target_folder)
#
# # 遍历源文件夹
# for root, dirs, files in os.walk(source_folder):
#     for file in files:
#         file_path = os.path.join(root, file)
#         # 检查文件名是否以'det'、'dig'或'lig'开头
#         if file.startswith(('det', 'dig', 'lig')):
#             # 构建目标文件路径
#             target_file_path = os.path.join(target_folder, file)
#             # 移动文件到目标文件夹
#             shutil.move(file_path, target_file_path)
#             print(f"移动文件: {file_path} 到 {target_file_path}")
#
# print("移动完成。")

# import os
# import json
#
# # 定义一个函数，检查标注信息是否是四点标注
# def is_four_point_annotation(annotation):
#     return len(annotation["points"]) == 4
#
# # 设置文件夹路径
# folder_path = "/home/zhihui/Dataset/data_poi_921/label"
#
# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     if filename.endswith(".json"):
#         # 构建完整的文件路径
#         file_path = os.path.join(folder_path, filename)
#
#         # 打开JSON文件并解析内容
#         with open(file_path, 'r') as json_file:
#             data = json.load(json_file)
#
#             # 假设标注信息存储在 'shapes' 键下
#             shapes = data.get('shapes', [])
#
#
#             # 检查每个形状的标注信息是否都是四点标注
#             all_four_point = all(is_four_point_annotation(shape) for shape in shapes)
#
#             # 如果不是四点标注，则打印文件名
#             if not all_four_point:
#                 print(f"File '{filename}' contains non-four-point annotations.")


# import os
# import json
#
# # 定义一个函数，检查标注信息是否是四点标注
# def is_four_point_annotation(annotation):
#     return len(annotation["points"]) == 4
#
# # 设置文件夹路径
# folder_path = "/home/zhihui/Dataset/data_poi_921/label"
#
# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     if filename.endswith(".json"):
#         # 构建完整的文件路径
#         file_path = os.path.join(folder_path, filename)
#
#         # 打开JSON文件并解析内容
#         with open(file_path, 'r') as json_file:
#             data = json.load(json_file)
#
#             # 假设标注信息存储在 'shapes' 键下
#             shapes = data.get('shapes', [])
#
#             # 初始化标志变量
#             has_non_four_point_annotation = False
#
#             # 遍历标注信息
#             for shape in shapes:
#                 label = shape.get("label", "")
#                 if label in ["poi_val0", "poi_val1"]:
#                     # 检查标注信息是否是四点标注
#                     if not is_four_point_annotation(shape):
#                         # 如果不是四点标注，则设置标志变量为 True
#                         has_non_four_point_annotation = True
#                         print(f"File '{filename}' contains non-four-point annotation for label '{label}'.")
#
#             # 如果标志变量为 True，表示该文件包含不符合要求的标注
#             if has_non_four_point_annotation:
#                 # 在这里可以执行相应的操作
#                 pass



# import os
# import json
#
# # 指定要遍历的文件夹路径
# folder_path = '/home/zhihui/Dataset/data_poi_926/label'
#
# # 遍历文件夹中的文件
# for filename in os.listdir(folder_path):
#     if filename.endswith('.json'):
#         # 构建完整的文件路径
#         file_path = os.path.join(folder_path, filename)
#
#         # 打开JSON文件并解析内容
#         with open(file_path, 'r') as json_file:
#             data = json.load(json_file)
#
#         # 标记是否存在空的description或description不存在
#         has_empty_description = False
#
#         # 检查 "shapes" 字段是否存在
#         if 'shapes' in data:
#             # 遍历 "shapes" 列表
#             for shape in data['shapes']:
#                 label = shape.get('label', '')
#
#                 # 检查 "label" 是否为 "poi_val0" 或 "poi_val1"
#                 if label in ['poi_val0', 'poi_val1']:
#                     # 检查 "description" 字段是否存在并且为空
#                     if 'description' not in shape or not shape['description']:
#                         has_empty_description = True
#                         break  # 如果找到空的description，不再继续检查其他shapes
#
#         # 如果存在空的description或description不存在
#         if has_empty_description:
#             # 打印文件名
#             print(f'File: {filename} - Description in "poi_val0" or "poi_val1" is empty or does not exist.')



# import os
# import re
# import shutil
#
# # 定义图片文件夹路径
# image_folder = '/home/zhihui/Dataset/data_poi_109/images'
#
# # 获取图片文件夹中的所有文件名
# image_files = os.listdir(image_folder)
#
# # 定义txt文件路径
# txt_file = '/home/zhihui/Downloads/work-zc/name.txt'
#
# # 用于存储文件名
# txt_filenames = []
#
# # 读取txt文件中的内容
# with open(txt_file, 'r') as file:
#     lines = file.readlines()
#
# # 提取文本文件中的文件名
# for line in lines:
#     # 使用正则表达式来提取文件名，例如：'images 672.jpg ['0', '20']'
#     match = re.search(r'images (\S+)', line)
#     if match:
#         filename = match.group(1)
#         print(filename)
#         txt_filenames.append(filename)
#
# # 检查哪些文件名在图片文件夹中没有出现
# missing_files = [filename for filename in txt_filenames if filename not in image_files]
#
# # 获取最后的46个文件名
# last_46_filenames = txt_filenames[-46:]
#
# # 打印最后46个文件名
# for filename in last_46_filenames:
#     print(f"File to be moved: {filename}")
#
# target_directory = '/home/zhihui/Dataset/data_poi_109/temp'
# # 移动文件到目标目录
# for filename in last_46_filenames:
#     source_path = os.path.join(image_folder, filename)
#     if os.path.exists(source_path):
#         target_path = os.path.join(target_directory, filename)
#         shutil.move(source_path, target_path)
#         print(f"Moved file to target directory: {filename}")
#     else:
#         print(f"File not found in image folder: {filename}")
#
# # 打印没有出现在图片文件夹中的文件名
# for missing_file in missing_files:
#     print(f"File not found in image folder: {missing_file}")

# import os
# import shutil
#
# # 指定要遍历的文件夹路径
# folder_path = '/home/zhihui/Dataset/csg_result_0'
# folder_path2 = '/home/zhihui/Dataset/data_2/images'
#
# # 指定要保存文件名的txt文件
# output_file = 'image_filenames.txt'
#
# # 支持的图片文件扩展名
# image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
#
# # 遍历文件夹并写入文件名到txt文件
# with open(output_file, 'w') as txt_file:
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             # 检查文件扩展名是否为图片格式
#             if any(file_path.lower().endswith(ext) for ext in image_extensions):
#                 txt_file.write(file + '\n')
#
# print(f'Image filenames have been written to {output_file}')
#
#
# # 新建目标文件夹的路径
# new_folder_path = '/home/zhihui/Dataset/csg_result_0/get'
#
# # 创建用户指定的目标文件夹
# os.makedirs(new_folder_path, exist_ok=True)
#
# # 读取txt文件中的文件名
# with open('image_filenames.txt', 'r') as txt_file:
#     image_filenames = txt_file.read().splitlines()
#
# # 遍历文件名列表并复制文件到新文件夹
# for filename in image_filenames:
#     source_file = os.path.join(folder_path2, filename)
#     if os.path.isfile(source_file):
#         destination_file = os.path.join(new_folder_path, filename)
#         shutil.copy(source_file, destination_file)
#
# print(f'Files from image_filenames.txt have been copied to {new_folder_path}')

# ##删除像素较小的部分图像
# import os
# from pathlib import Path
# from typing import List
#
# def get_file_size(file_path: Path) -> int:
#     """获取文件大小"""
#     return file_path.stat().st_size
#
# def get_sorted_files(folder_path: Path) -> List[Path]:
#     """获取文件夹内所有文件，并按文件大小排序"""
#     files = list(folder_path.glob('*'))
#     files.sort(key=get_file_size)
#     return files
# def remove_top_n_files(files: List[Path], n: int) -> None:
#     for file in files[:n]:
#         file.unlink()
#
# if __name__ == '__main__':
#     folder_path = Path('/home/zhihui/Dataset/lig_cut/image')
#     files = get_sorted_files(folder_path)
#     remove_top_n_files(files, 500)

import os
from pathlib import Path

img_folder = Path('/home/zhihui/Dataset/lig_cut/image')
label_folder = Path('/home/zhihui/Dataset/lig_cut/label')

img_files = list(img_folder.glob('*.jpg')) + list(img_folder.glob('*.png'))+ list(img_folder.glob('*.jpeg'))
label_files = list(label_folder.glob('*.txt'))

# 获取所有文件名(不含扩展名)
img_names = [str(path.stem) for path in img_files]
label_names = [str(path.stem) for path in label_files]

# 查找不匹配的文件名
to_delete = []
for name in img_names:
    if name not in label_names:
        to_delete.append(name)
for name in label_names:
    if name not in img_names:
        to_delete.append(name)

# 删除不匹配的文件
for name in to_delete:
    img_file = img_folder / (name + '.jpg')
    label_file = label_folder / (name + '.txt')

    if img_file.exists():
        img_file.unlink()
    if label_file.exists():
        label_file.unlink()





