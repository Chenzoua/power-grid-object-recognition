#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import numpy as np
# from util import strs
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
# from util.io import read_lines
# from util.misc import norm2
import json
import os
import cv2
from util.augmentation import Augmentation
from util import canvas as cav
import time
from dataset.meter_data import Meter
from dataset.ocrdata import OCR_data

if __name__ == '__main__':


    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640, mean=means, std=stds
    )

    trainset = Meter(transform=transform)
    
    valid_samples = []


    for idx in range(0, len(trainset)):
        t0 = time.time()
        # print("idx",idx)


        img, pointer_mask, dail_mask, text_mask, train_mask, bboxs0, bboxs1, transcripts = trainset[idx]
        
         # 根据您的过滤条件判断样本是否有效
        if transcripts and len(bboxs0) > 0 and len(bboxs1) > 0:
            valid_samples.append((img, pointer_mask, dail_mask, text_mask, train_mask, bboxs0, bboxs1, transcripts))
        else:
            # 打印错误样本信息
            print("Error: Sample with index {} is invalid.".format(idx))
            print("Transcripts:", transcripts)
            print("Bboxs0:", bboxs0)
            print("Bboxs1:", bboxs1)


#         img = img.transpose(1, 2, 0)
#         img = ((img * stds + means) * 255).astype(np.uint8)

        print("trans",transcripts)
        # print("bboxs",bboxs0,bboxs1)
#         cv2.imshow('imgs', img)
        # cv2.imshow("pointer_mask", cav.heatmap(np.array(pointer_mask * 255 / np.max(pointer_mask), dtype=np.uint8)))
        # cv2.imshow("dail_mask", cav.heatmap(np.array(dail_mask * 255 / np.max(dail_mask), dtype=np.uint8)))
#         cv2.imshow("text_mask", cav.heatmap(np.array(text_mask * 255 / np.max(text_mask), dtype=np.uint8)))
#         cv2.waitKey(0)

