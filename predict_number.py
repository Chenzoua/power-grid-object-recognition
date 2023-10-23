import time
import cv2
import os
from get_number_area import Detector_nuber
# from PIL import Image

det_nuber=Detector_nuber()

from paddleocr import PaddleOCR, draw_ocr

ocr = PaddleOCR(lang='en')
predict_dir = 'demo/'
image_list = os.listdir(predict_dir)
start_time = time.time()
for index in image_list:
    print("**************", index)
    images = cv2.imread(predict_dir + index)
    cv2.namedWindow("index", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("index", 800, 600)
    cv2.imshow("index", images)
    cv2.waitKey(0)
    image, image_info, number_list= det_nuber.detect(images, index)

    if len(number_list) == 0:
        print("no detected number")
        continue
    else:
        for i in number_list:
            # cv2.imshow("index", i)
            # cv2.waitKey(0)
            # cv2.imwrite("pre_exp/number.jpg", i)
            ocr_res = ocr.ocr(i, cls=True, det=False)
            for line in ocr_res:
                print(line)

            txts = [line[0][0] for line in ocr_res]  # 直接获取文本部分
            print(txts)
            end_time = time.time()
            execution_time = end_time - start_time
            print("模型测试时间：", execution_time, "秒")

            font = cv2.FONT_HERSHEY_SIMPLEX
            result_img = cv2.putText(images, str(txts), (200, 200), font, 6, (0, 255, 0), 9)  # 仪表读数图像输出
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("result", 800, 600)
            cv2.imshow("result", result_img)
            cv2.waitKey(0)
            # cv2.imwrite("pre_exp/result_dig.jpg", result_img)
            cv2.destroyAllWindows()













