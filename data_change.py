# import os
# import json
#
#
# def labelme_to_yolo(labelme_json_path, yolo_txt_path, class_mapping):
#     with open(labelme_json_path, 'r') as f:
#         labelme_data = json.load(f)
#
#     with open(yolo_txt_path, 'w') as f:
#         for shape in labelme_data['shapes']:
#             label = shape['label']
#             points = shape['points']
#             x_center = (points[0][0] + points[2][0]) / 2
#             y_center = (points[0][1] + points[2][1]) / 2
#             width = abs(points[0][0] - points[2][0])
#             height = abs(points[0][1] - points[2][1])
#
#             class_id = class_mapping.get(label, -1)
#             if class_id == -1:
#                 continue
#
#             yolo_format = f"{class_id} {x_center} {y_center} {width} {height}\n"
#             f.write(yolo_format)
#
#
# def process_folder(folder_path, output_folder, class_mapping):
#     for root, _, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith('.json'):
#                 json_path = os.path.join(root, file)
#                 yolo_txt_path = os.path.join(output_folder, file.replace('.json', '.txt'))
#                 labelme_to_yolo(json_path, yolo_txt_path, class_mapping)
#
#
# if __name__ == "__main__":
#     input_folder = "/home/zhihui/Downloads/work-zc/data_copy/M2021/"
#     output_folder = "/home/zhihui/Downloads/Detect-and-read-meters-2/data_M2021/labels/train/"
#     class_mapping = {"number": 0}  # Update with your class mapping
#
#     process_folder(input_folder, output_folder, class_mapping)

#-----------------------------------------------------------------------------------------------------------------

#像素值坐标数据集转换
import os
import json
import cv2

#
# def labelme_to_yolo(labelme_data, img_width, img_height):
#     yolo_labels = []
#
#     for shape in labelme_data['shapes']:
#         if len(shape['points']) != 4:
#             continue
#
#         x_coords = [p[0] for p in shape['points']]
#         y_coords = [p[1] for p in shape['points']]
#
#         x_min = min(x_coords)
#         y_min = min(y_coords)
#         x_max = max(x_coords)
#         y_max = max(y_coords)
#
#         x_center = (x_min + x_max) / 2
#         y_center = (y_min + y_max) / 2
#         width = x_max - x_min
#         height = y_max - y_min
#
#         class_id = shape['label']
#         yolo_labels.append(
#             f"{class_id} {x_center / img_width} {y_center / img_height} {width / img_width} {height / img_height}\n")
#
#     return yolo_labels
#
#
# def process_folder(input_folder, output_folder):
#     for root, _, files in os.walk(input_folder):
#         for file in files:
#             if file.lower().endswith('.json'):
#                 json_path = os.path.join(root, file)
#                 output_path = os.path.join(output_folder, file.replace('.json', '.txt'))
#
#                 with open(json_path, 'r') as f:
#                     labelme_data = json.load(f)
#
#                 image_filename = labelme_data['imagePath']
#                 image_path = os.path.join(root, image_filename)
#                 image = cv2.imread(image_path)
#                 print(image.shape)
#                 img_height, img_width, _ = image.shape
#                 yolo_labels = labelme_to_yolo(labelme_data, img_width, img_height)
#
#                 with open(output_path, 'w') as f:
#                     f.writelines(yolo_labels)
#
#
# if __name__ == "__main__":
#     input_folder = "/home/zhihui/Downloads/work-zc/data_copy/M2021/anno_M2021/"
#     output_folder = "/home/zhihui/Downloads/Detect-and-read-meters-2/data_M2021/labels/train/"
#     class_mapping = {"number": 0}  # Update with your class mapping
#
#     process_folder(input_folder, output_folder)

# -----------------------------------------------------------------------------------------------------------------
# josn2yolo
import os
import json

# 定义类别标签映射，将标签名称映射到整数标签
class_label_mapping = {
    # "dig_num": 0,

    "dig": 0,
    "det": 1,
    "lig_off":2,
    "lig_on":3,

    # 添加更多类别映射
}

# 输入文件夹路径包含JSON文件和图像文件
input_folder = "/home/zhihui/Dataset/singapore/data"
output_folder = "/home/zhihui/Dataset/singapore/data/label"

# 遍历文件夹中的JSON文件
for json_filename in os.listdir(input_folder):
    print(json_filename)
    if json_filename.endswith(".json"):
        json_path = os.path.join(input_folder, json_filename)

        # 从JSON文件名获取对应的txt文件名
        txt_filename = json_filename.replace(".json", ".txt")
        yolo_path = os.path.join(output_folder, txt_filename)

        # 读取JSON文件
        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        # 获取图像的宽度和高度
        image_width = data["imageWidth"]
        image_height = data["imageHeight"]

        with open(yolo_path, "w") as yolo_file:
            # 遍历JSON中的形状（标注）
            for shape in data["shapes"]:
                label = shape["label"]
                points = shape["points"]

                # 计算YOLO格式的坐标
                x_center = (points[0][0] + points[1][0]) / (2 * image_width)
                y_center = (points[0][1] + points[1][1]) / (2 * image_height)
                width = abs(points[1][0] - points[0][0]) / image_width
                height = abs(points[1][1] - points[0][1]) / image_height

                # 获取类别标签的整数编码
                class_label = class_label_mapping[label]

                # 写入YOLO格式的行（格式：class x_center y_center width height）
                yolo_line = f"{class_label} {x_center} {y_center} {width} {height}\n"
                yolo_file.write(yolo_line)

print("Conversion completed.")


