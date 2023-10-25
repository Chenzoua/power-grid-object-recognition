# import cv2
# import os
#
# def get_image_dimensions(image_path):
#     image = cv2.imread(image_path)
#     height, width, _ = image.shape
#     return width, height
#
# def process_folder(folder_path):
#     for root, _, files in os.walk(folder_path):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 image_path = os.path.join(root, file)
#                 width, height = get_image_dimensions(image_path)
#                 print(f"Image: {file}, Width: {width}, Height: {height}")
#
# if __name__ == "__main__":
#     input_folder = "/home/zhihui/Downloads/work-zc/data_copy/M2021/"
#     process_folder(input_folder)

import os
import matplotlib
matplotlib.use('TkAgg')
import cv2



def visualize_images_with_bbox(image_folder, label_folder, class_names):
    for image_filename in os.listdir(image_folder):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
            image_path = os.path.join(image_folder, image_filename)
            label_filename = image_filename.replace('.jpg', '.txt').replace('.png', '.txt').replace('.JPG', '.txt')  # 构建对应的标签文件名
            label_path = os.path.join(label_folder, label_filename)

            image = cv2.imread(image_path)
            print(image_filename)
            h, w, _ = image.shape

            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.split())

                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)

                    class_name = class_names[int(class_id)]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # 创建一个可调整大小的窗口
            # cv2.resizeWindow("Image", 1000, 800)  # 设置窗口大小为800x600像素
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



if __name__ == "__main__":
    image_folder = "/home/zhihui/Dataset/lig_cut/image"
    label_folder = "/home/zhihui/Dataset/lig_cut/label"
    class_names = ["dig", "det", "lig_off", "lig_on"]  # 替换为你的类别名称列表

    visualize_images_with_bbox(image_folder, label_folder, class_names)
