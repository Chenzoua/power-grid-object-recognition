import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network.vgg import VggNet
from network.resnet import ResNet
from util.roi import batch_roi_transform
from network.crnn import CRNN
from util.converter import keys
from util.misc import mkdirs, to_device
import cv2
from util.tool import order_points
# import paddleocr

class Recognizer(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        self.crnn = CRNN(32, 1, nclass, 256)

    def forward(self, rois):
        return self.crnn(rois)


class OCRnet(nn.Module):
    def __init__(self, backbone='vgg', is_training=True):
        super().__init__()

        self.is_training = is_training
        self.backbone_name = backbone

        # ##class and regression branch
        self.out_channel = 1
        self.predict = nn.Sequential(
            nn.Conv2d(32, self.out_channel, kernel_size=1, stride=1, padding=0)
        )

        num_class = len(keys) + 1
        self.recognizer = Recognizer(num_class)


    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict['model'])

    def forward(self, x,boxes,mapping):
        # x_out = self.backbone_name(x)
        rois = batch_roi_transform(x, boxes[:, :8], mapping)
        pred_mapping = mapping
        # 将输入的映射信息存储在pred_mapping中以备后续使用
        pred_boxes = boxes
        # 将输入的边界框信息存储在pred_boxes中，以备后续使用

        print("rois",rois.shape)
        preds = self.recognizer(rois)
        # print("preds",preds.shape)

        preds_size = torch.LongTensor([preds.size(0)] * int(preds.size(1)))
        preds_size=to_device(preds_size)
        # print("predsize", preds_size)

        return (preds, preds_size)

    def forward_test2(self, x):
        up1, up2, up3, up4, up5 = self.fpn(x)
        output = self.predict(up1)
        # print("predict_out",output.shape)

        text_pred = torch.sigmoid(output[0, 0, :, :]).data.cpu().numpy()
        text_pred = (text_pred > 0.5).astype(np.uint8)

        text_label = self.filter(text_pred)

        # cv2.imshow("srtc", text_pred * 255)
        # cv2.waitKey(0)
        # cv2.imwrite("pre_exp/2.jpg", text_pred * 255)
        # new
        text_edges = text_label * 255
        text_edges = text_edges.astype(np.uint8)
        text_contours,_ = cv2.findContours(text_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # _, text_contours, _ = cv2.findContours(text_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ref_point = []
        for i in range(len(text_contours)):
            rect = cv2.minAreaRect(text_contours[i])
            ref_point.append((int(rect[0][0]), int(rect[0][1])))
        # print("ref",ref_point)

        word_edges = text_label * 255
        # img_bin, contours, hierarchy = cv2.findContours(word_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(word_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_dis = 10000
        index = 0
        if len(contours) != 0:
            # for i in range(len(contours)):
            #     min_rect = cv2.minAreaRect(contours[i])
            #
            #     test_point = (min_rect[0][0], min_rect[0][1])
            #     dis = (test_point[0] - std_point[1][0]) ** 2 + (test_point[1] - std_point[1][1]) ** 2
            #     if dis < max_dis:
            #         max_dis = dis
            #         index = i

            rect_points = cv2.boxPoints(cv2.minAreaRect(contours[index]))
            bboxes = np.int0(rect_points)
            bboxes = order_points(bboxes)
            # print("bbox", bboxes)
            boxes = bboxes.reshape(1, 8)
            mapping = [0]
            mapping = np.array(mapping)
            rois = batch_roi_transform(x, boxes[:, :8], mapping)
            # print("rois",rois.shape)
            preds = self.recognizer(rois)
            preds_size = torch.LongTensor([preds.size(0)] * int(preds.size(1)))
            # print("*******", preds.shape, preds_size)

        else:
            preds = None
            preds_size = None

        return text_label, (preds, preds_size),bboxes


    def filter(self,image,n=30):
        text_num, text_label = cv2.connectedComponents(image.astype(np.uint8), connectivity=8)
        for i in range(1, text_num + 1):
            pts = np.where(text_label == i)
            if len(pts[0]) < n:
                text_label[pts] = 0
        text_label = text_label > 0
        text_label = np.clip(text_label, 0, 1)
        text_label = text_label.astype(np.uint8)
        return text_label






