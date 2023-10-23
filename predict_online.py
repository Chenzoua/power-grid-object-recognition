import os
import cv2
import numpy as np
import time
from util.augmentation import BaseTransform
from util.config import config as cfg, update_config, print_config
from util.option import BaseOptions
from network.textnet import TextNet
from util.detection_mask import TextDetector as TextDetector_mask
import torch
from util.misc import to_device
from util.read_meter import MeterReader
from util.converter import keys,StringLabelConverter
from get_meter_area import  Detector
from get_number_area import Detector_nuber

# parse arguments

option = BaseOptions()
args = option.initialize()
update_config(cfg, args)
# print_config(cfg)

predict_dir='demo/' ##预测目录
model = TextNet(is_training=True, backbone=cfg.net)
model_path = os.path.join(cfg.eval_dir,
                          'textgraph_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))
print(model_path)
model.load_model(model_path)
model = model.to(cfg.device)
# print(model)
converter=StringLabelConverter(keys)

det=Detector()
det_nuber=Detector_nuber()
# print(det)
detector = TextDetector_mask(model)
meter = MeterReader()
transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)

from paddleocr import PaddleOCR, draw_ocr
ocr = PaddleOCR(lang='en')


start_time = time.time()
image_list=os.listdir(predict_dir)
print(image_list)

# start_time = time.time()

for index in image_list:
    print("**************",index)
    image = cv2.imread(predict_dir+index)
    # cv2.imshow("det1",image)
    # cv2.waitKey(0)
    # cv2.imwrite("pre_exp/det1.jpg", image)
    # detect meter area
    image, image_info, digital_list, meter_list, obj_res = det.detect(image, index)
    print(obj_res)

    # cv2.imshow("test", image)
    # cv2.waitKey(0)

    if len(digital_list) != 0:
        for dig in digital_list:
            # cv2.imshow('dig',dig)
            # cv2.waitKey(0)
            # cv2.imwrite("pre_exp/dig_res.jpg",dig)
            image,  number_list = det_nuber.detect(dig)
            if len(number_list) == 0:
                print("no detected number")
                continue
            else:
                for i in number_list:
                    # cv2.imshow("index", i)
                    # cv2.waitKey(0)
                    # cv2.imwrite("pre_exp/number.jpg", i)
                    ocr_res = ocr.ocr(i, cls=True, det=False)
                    # for line in ocr_res:
                    #     print(line)

                    txts = [line[0][0] for line in ocr_res]  # 直接获取文本部分
                    for idx, txt in enumerate(txts):
                        if txt.count('.') >= 2:
                            txts[idx] = txt.replace('.', '')
                    print(txts)

    if len(meter_list)!=0:
        # print("----------------------------------------------------------------------------------------------------------------------")
        for i in meter_list:
            # cv2.imshow("det",i)
            # cv2.waitKey(0)
            # cv2.imwrite("pre_exp/det.jpg", i)
            image,_=transform(i)
            image = image.transpose(2, 0, 1)
            image=torch.from_numpy(image).unsqueeze(0)
            image=to_device(image)
            output = detector.detect1(image)
            pointer_pred, dail_pred, text_pred, preds1, preds2, std_points = output['pointer'], output['dail'], output['text'], output['reco1'], output['reco2'], output['std']

            img_show = image[0].permute(1, 2, 0).cpu().numpy()
            img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

            result = meter(img_show,pointer_pred, dail_pred, text_pred, preds1, preds2, std_points)
            # print('resu',result)
    else:
        print("no detected")

end_time = time.time()
execution_time = end_time - start_time
print("模型测试时间：", execution_time, "秒")



