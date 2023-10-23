
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
        raw_image = cv2.imread(image_dir)
        out_image = []
        if len(init_bbox) >= 4:
            x1 = int(init_bbox[0])
            x2 = int(init_bbox[2])
            y1 = int(init_bbox[1])
            y2 = int(init_bbox[3])
            image = raw_image[y1:y2,x1:x2]
            out_image = raw_image[y1:y2,x1:x2]
        else:
            image = raw_image
            out_image = raw_image
        image, image_info, digital_list, meter_list, obj_res = self.det.detect(image, 0)
        return out_image, obj_res

    def read(self, image, detect_result, output_dir):
        print("READ")
        read_result_list = []
        for i in range(len(detect_result)):
            print("read " + str(i))
            read_type = detect_result[i][0]
            read_bbox = detect_result[i][1:5]

            data = [read_type, read_bbox]
            x1 = int(read_bbox[0])
            x2 = int(read_bbox[2])
            y1 = int(read_bbox[1])
            y2 = int(read_bbox[3])
            bbox_image = image[y1:y2,x1:x2]
            value_list = []
            image_list = []
            if read_type == 0:
                if output_dir != "":
                    bbox_image_path = output_dir + "/det_" + str(i) + ".jpg"
                    cv2.imwrite(bbox_image_path, bbox_image)
                value_list, image_list = self.read_det(bbox_image, cfg)
            elif read_type == 1:
                if output_dir != "":
                    bbox_image_path = output_dir + "/dig_" + str(i) + ".jpg"
                    cv2.imwrite(bbox_image_path, bbox_image)
                value_list, image_list = self.read_dig(bbox_image)
            elif read_type == 2:
                if output_dir != "":
                    bbox_image_path = output_dir + "/lig_off_" + str(i) + ".jpg"
                    cv2.imwrite(bbox_image_path, bbox_image)
            elif read_type == 3:
                if output_dir != "":
                    bbox_image_path = output_dir + "/lig_on_" + str(i) + ".jpg"
                    cv2.imwrite(bbox_image_path, bbox_image)
            read_result_list.append([i, read_type, read_bbox, value_list, image_list])
        return read_result_list
    def read_det(self, image, cfg):
        print("read_det")
        value_list = []
        image_list = []

        det_image, _ = self.transform(image)
        det_image = det_image.transpose(2, 0, 1)
        det_image = torch.from_numpy(det_image).unsqueeze(0)
        det_image = to_device(det_image)
        output = self.detector.detect1(det_image)

        pointer_pred, dail_pred, text_pred, preds1, preds2, std_points = output['pointer'], output['dail'], output['text'], output['reco1'], output['reco2'], output['std']

        # pointer_pred, dail_pred, text_pred, preds, std_points = output['pointer'], output['dail'], output['text'], output['reco'], output['std']

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
        #     meter_image = find_lines(img, pointer_pred, dail_pred, pred_transcripts, std_points, value)

        #     if value != None:
        #         value_list.append(value)
        #         image_list.append(meter_image)
        # return value_list, image_list
        img = det_image[0].permute(1,2,0).cpu().numpy()
        img = ((img * cfg.stds + cfg.means) * 255).astype(np.uint8)
        result = self.meter(img, pointer_pred, dail_pred, text_pred, preds1, preds2, std_points)
        meter_image = find_lines(img, pointer_pred, dail_pred, std_points, result)
        
        if result != None:
            value_list.append(result)
            image_list.append(meter_image)
        return value_list, image_list

    def read_dig(self, image):
        value_list = []
        image_list = []
        print("read_dig")
        result_image, result_list = self.det_nuber.detect(image)
        print(result_list)
        if len(result_list) != 0:
            for n in result_list:
                ocr_res = self.ocr.ocr(n, cls = True, det = False)
                txts = [line[0][0] for line in ocr_res]
                if len(txts) >= 1: # and txts[0].isdigit():
                    if self.is_number(txts[0]):
                        value = float(txts[0])
                        value_list.append(value)
                        image_list.append(result_image)

        return value_list, image_list
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

def draw_test_image(image, detect_result, read_result, output_dir):
    for i in range(len(detect_result)):
        read_type = detect_result[i][0]
        read_bbox = detect_result[i][1:5]

        x1 = int(read_bbox[0])
        x2 = int(read_bbox[2])
        y1 = int(read_bbox[1])
        y2 = int(read_bbox[3])
        if read_type == 0:
            if len(read_result[i]) >= 5 and read_result[i][4] != []:
                resized_read_result_image = cv2.resize(read_result[i][4][0], (image[y1:y2,x1:x2].shape[1], image[y1:y2,x1:x2].shape[0]), interpolation = cv2.INTER_AREA)
                image[y1:y2,x1:x2] = resized_read_result_image  
            cv_draw_box(image, read_bbox, "det", (0, 0, 255))
        elif read_type == 1:
            if len(read_result[i]) >= 5 and read_result[i][4] != []:
                resized_read_result_image = cv2.resize(read_result[i][4][0], (image[y1:y2,x1:x2].shape[1], image[y1:y2,x1:x2].shape[0]), interpolation = cv2.INTER_AREA)
                cv_draw_num(resized_read_result_image, read_result[i][3], read_result[i][2])
                image[y1:y2,x1:x2] = resized_read_result_image  
            cv_draw_box(image, read_bbox, "dig", (0, 255, 0))
        elif read_type == 2:
            cv_draw_box(image, read_bbox, "lig_off", (255, 0, 255))
        elif read_type == 3:
            cv_draw_box(image, read_bbox, "lig_on", (255, 255, 0))
        test_image_path = output_dir #  + "/test.jpg"
        cv2.imwrite(test_image_path, image)
def cv_draw_box(image, bbox, label, box_color=(0,0,255)):
    font = cv2.FONT_HERSHEY_SIMPLEX

    label_size = cv2.getTextSize(label, font, 1, 2)

    text_origin = np.array([bbox[0], bbox[1] - label_size[0][1]])

    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color = box_color, thickness = 1)
    cv2.rectangle(image, tuple(text_origin), tuple(text_origin + label_size[0]), color = box_color, thickness = -1)
    cv2.putText(image, label, (bbox[0], bbox[1] - 5), font, 1, (0,0,0), 1)
def cv_draw_num(image, num_list, bbox):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(num_list)):
        cv2.putText(image, str(int(num_list[i])), (0, 25 + 25 * i), font, 1, (0,255,255), 3)

def find_lines(ori_img,pointer_mask,dail_mask,std_point, value):
    # 实施骨架算法
    from skimage import morphology
    pointer_skeleton = morphology.skeletonize(pointer_mask)
    pointer_edges = pointer_skeleton * 255
    pointer_edges = pointer_edges.astype(np.uint8)

    pointer_lines = cv2.HoughLinesP(pointer_edges, 1, np.pi / 180, 10, np.array([]), minLineLength=10,
                                    maxLineGap=400)
    try:
        for x1, y1, x2, y2 in pointer_lines[0]:
            coin1 = (x1, y1)
            coin2 = (x2, y2)
            cv2.line(ori_img, (x1, y1), (x2, y2), (255, 0, 255), 4)
    except TypeError:
        return None
    if std_point is None:
        return ori_img
    if len(std_point) > 0:
        a1 = std_point[0]
        cv2.circle(ori_img, a1, 4, (255, 0, 0), -1)
    if len(std_point) > 1:
        a2 = std_point[1]
        cv2.circle(ori_img, a2, 4, (255, 0, 0), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    ori_img = cv2.putText(ori_img, str(value), (30, 30), font, 1.2, (255, 0,255), 2)  # 仪表读数图像输出
    return ori_img

if __name__=="__main__":
    predict_dir = "/home/zhihui/ImageRecognitio/Detect-and-read-meters/data/det_test_2.jpg"
    output_dir = "/home/zhihui/ImageRecognitio/Detect-and-read-meters/data/result"

    time0 = time.time()
    image_detector = ImageDetector()
    time1 = time.time()
    raw_image, detect_result = image_detector.detect(predict_dir, [])
    time2 = time.time()
    print(detect_result)
    read_result_list = image_detector.read(raw_image, detect_result, output_dir)
    time3 = time.time()
    print(read_result_list)

    test_image = cv2.imread(predict_dir)
    draw_test_image(test_image, detect_result, read_result_list, output_dir + "/test.jpg")

    print("load time " + str(time1 - time0))
    print("detect time " + str(time2 - time1))
    print("read time " + str(time3 - time2))
