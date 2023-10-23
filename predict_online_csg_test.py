import os
import time
import shutil
import re
import sys
import cv2

from shutil import copyfile

from predict_online_test import ImageDetector, draw_test_image

def find_file_name_list(data_path):
    data_filename_list = []
    root_dirs_files = os.walk(data_path)
    while True:
        try:
            iter_path = root_dirs_files.__next__()
            file_dir = iter_path[0]
            next_file_dir = iter_path[1]
            file_list = iter_path[2]
            if len(file_list) != 0:
                for file_name in file_list:
                    file_name_str_list = file_name.split(".")
                    file_name_ext = file_name_str_list[-1]
                    if file_name_ext == "png" or file_name_ext == "jpg" or file_name_ext == "jpeg":
                        data_filename_list.append([file_dir, file_name])
        except StopIteration:
            break
    return data_filename_list 
if __name__=="__main__":
    predict_dir = "/home/zhihui/ImageRecognitio/power-grid-object-recognition/data/csg_test"
    output_dir = "/home/zhihui/ImageRecognitio/power-grid-object-recognition/data/csg_result"

    filename_list = find_file_name_list(predict_dir)
    file_num = len(filename_list)

    image_detector = ImageDetector()
    p = 0
    for file_data in filename_list:
        file_dir = file_data[0]
        file_name = file_data[1]

        predict_file = file_dir + "/" + file_name

        raw_image, detect_result = image_detector.detect(predict_file, [])
        read_result_list = image_detector.read(raw_image, detect_result, "")

        test_image = cv2.imread(predict_file)
        draw_test_image(test_image, detect_result, read_result_list, output_dir + "/" + file_name)
        p += 1
        print(str(p) + " / " + str(file_num))
        print(file_name)
