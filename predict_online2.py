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
import re


class ImageDetector:
    def __init__(self):
        option = BaseOptions()
        args = option.initialize()
        update_config(cfg, args)

        self.model = TextNet(is_training=True, backbone=cfg.net)
        # model_path = os.path.join(cfg.save_dir, cfg.exp_name,
        #                           'textgraph_{}_{}.pth'.format(self.model.backbone_name, cfg.checkepoch))
        model_path = os.path.join(cfg.eval_dir,
                                  'textgraph_{}_{}.pth'.format("resnet50", "300"))
        self.model.load_model(model_path)
        self.model = self.model.to(cfg.device)
        self.converter=StringLabelConverter(keys)
        
        from get_meter_area import  Detector
        from get_number_area import Detector_nuber

        self.det = Detector() # yolov5
        self.det_nuber = Detector_nuber() # dig number detector
        self.detector = TextDetector_mask(self.model) # meter poi detector
        self.meter = MeterReader() # meter value
        self.transform = BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)

        from paddleocr import PaddleOCR, draw_ocr
        self.ocr = PaddleOCR(lang='en')

    def detect(self, image_dir, init_bbox):
        print("DETECT")
        image = cv2.imread(image_dir)
        if len(init_bbox) >= 4:
            x1 = int(init_bbox[0])
            x2 = int(init_bbox[2])
            y1 = int(init_bbox[1])
            y2 = int(init_bbox[3])
            image = image[y1:y2,x1:x2]
        image, image_info, digital_list, meter_list, obj_res = self.det.detect(image, 0)
        return image, obj_res

    def read(self, image, detect_result, output_dir):
        print("READ")
        read_result_list = []
        for i in range(len(detect_result)):
            read_type = detect_result[i][0]
            read_bbox = detect_result[i][1:5]

            data = [read_type, read_bbox]
            x1 = int(read_bbox[0])
            x2 = int(read_bbox[2])
            y1 = int(read_bbox[1])
            y2 = int(read_bbox[3])
            bbox_image = image[y1:y2,x1:x2]
            value_list = []
            if read_type == 0:
                bbox_image_path = output_dir + "/det_" + str(i) + ".jpg"
                cv2.imwrite(bbox_image_path, bbox_image)
                value_list = self.read_det(bbox_image, cfg)
            elif read_type == 1:
                bbox_image_path = output_dir + "/dig_" + str(i) + ".jpg"
                cv2.imwrite(bbox_image_path, bbox_image)
                value_list = self.read_dig(bbox_image)
            elif read_type == 2:
                bbox_image_path = output_dir + "/lig_off_" + str(i) + ".jpg"
                cv2.imwrite(bbox_image_path, bbox_image)
            elif read_type == 3:
                bbox_image_path = output_dir + "/lig_on_" + str(i) + ".jpg"
                cv2.imwrite(bbox_image_path, bbox_image)
            read_result_list.append([i, read_type, read_bbox, value_list])
        return read_result_list
    def read_det(self, image, cfg):
        print("read_det")
        value_list = []

        det_image, _ = self.transform(image)
        det_image = det_image.transpose(2, 0, 1)
        det_image = torch.from_numpy(det_image).unsqueeze(0)
        det_image = to_device(det_image)
        output = self.detector.detect1(det_image)

        # pointer_pred, dail_pred, text_pred, preds, std_points = output['pointer'], output['dail'], output['text'], output['reco'], output['std']
        pointer_pred, dail_pred, text_pred, preds1, preds2, std_points = output['pointer'], output['dail'], output['text'], output['reco1'], output['reco2'], output['std']

        # pred, preds_size = preds

        # if pred != None:
        #     _, pred = pred.max(2)
        #     pred = pred.transpose(1, 0).contiguous().view(-1)
        #     pred_transcripts = self.converter.decode(pred.data, preds_size.data, raw = False)
        #     pred_transcripts = [pred_transcripts] if isinstance(pred_transcripts, str) else pred_transcripts

        # else:
        #     pred_transcripts = None
        #     print("read meter failed!")

        # if pred_transcripts != None:
        #     img = det_image[0].permute(1, 2, 0).cpu().numpy()
        #     img = ((img * cfg.stds + cfg.means) * 255).astype(np.uint8)
        #     value = self.meter(img, pointer_pred, dail_pred, text_pred, pred_transcripts, std_points)
        #     if value != None:
        #         value_list.append(value)
        # return value_list
        img = det_image[0].permute(1,2,0).cpu().numpy()
        img = ((img * cfg.stds + cfg.means) * 255).astype(np.uint8)
        result = self.meter(img, pointer_pred, dail_pred, text_pred, preds1, preds2, std_points)
        if result != None:
            value_list.append(result)
        return value_list

    def read_dig(self, image):
        value_list = []
        print("read_dig")
        result_image, result_list = self.det_nuber.detect(image)
        if len(result_list) != 0:
            for n in result_list:
                ocr_res = self.ocr.ocr(n, cls = True, det = False)
                txts = [line[0][0] for line in ocr_res]
                if len(txts) >= 1: # and txts[0].isdigit():
                    if self.is_number(txts[0]):
                        value = float(txts[0])
                        value_list.append(value)
        return value_list
    def output(self, type_str, read_result_list):
        if type_str == "1:1:1:2:1:1":
            print("det")
            output_list = self.find_type(read_result_list, 0)
            if output_list == []:
                return []
            return output_list[-1]
        elif type_str == "1:1:1:2:1:2":
            print("dig")
            output_list = self.find_type(read_result_list, 1)
            if output_list == []:
                return []
            return output_list[-1]
        elif type_str == "1:1:1:2:3":
            print("lig")
            lig_off_list = self.find_type(read_result_list, 2)
            lig_on_list = self.find_type(read_result_list, 3)
            if lig_off_list != []:
                return [0]
            elif lig_on_list != []:
                return [1]
            else:
                return []
        else:
            return []

    def find_type(self, result_list, type_int):
        for x in result_list:
            if x[1] == type_int:
                return x
        return []
    def is_number(self, input_str):                                                       
        pattern = re.compile(r'^-?(([0-9]*(\.[0-9]{1,9})$)|([0-9]+$))')
        result = pattern.match(input_str)
        if result:
            return True
        else:
            return False

if __name__=="__main__":
    predict_dir = "/home/zhihui/ImageRecognitio/power-grid-object-recognition/data/dig128.jpg"
    output_dir = "/home/zhihui/ImageRecognitio/power-grid-object-recognition/data/result"

    time0 = time.time()
    image_detector = ImageDetector()
    time1 = time.time()
    raw_image, detect_result = image_detector.detect(predict_dir, [])
    time2 = time.time()
    print(detect_result)
    read_result_list = image_detector.read(raw_image, detect_result, output_dir)
    time3 = time.time()
    print(read_result_list)

    output_list = image_detector.output("1:1:1:2:1:1", read_result_list)
    print(output_list)
    output_list = image_detector.output("1:1:1:2:1:2", read_result_list)
    print(output_list)
    output_list = image_detector.output("1:1:1:2:3", read_result_list)
    print(output_list)

    print("load time " + str(time1 - time0))
    print("detect time " + str(time2 - time1))
    print("read time " + str(time3 - time2))
