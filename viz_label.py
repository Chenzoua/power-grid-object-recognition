#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
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
        size=480, mean=means, std=stds
    )

    trainset = OCR_data(transform=transform)


    for idx in range(0, len(trainset)):
        t0 = time.time()
        print("idx",idx)

        img, pointer_mask, dail_mask, text_mask, train_mask, bboxs1, transcripts = trainset[idx]
        # img,  text_mask, train_mask,  bboxs1, transcripts = trainset[idx]

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)

        # print('box0',bboxs0,'box1',bboxs1)
        print("trans",transcripts)
        cv2.imshow('imgs', img)
        # cv2.imshow("pointer_mask", cav.heatmap(np.array(pointer_mask * 255 / np.max(pointer_mask), dtype=np.uint8)))
        # cv2.imshow("dail_mask", cav.heatmap(np.array(dail_mask * 255 / np.max(dail_mask), dtype=np.uint8)))
        cv2.imshow("text_mask", cav.heatmap(np.array(text_mask * 255 / np.max(text_mask), dtype=np.uint8)))
        cv2.waitKey(0)

