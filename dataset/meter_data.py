#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
import json


class Meter(TextDataset):

    def __init__(self, root='/home/zhihui/Dataset/data_poi_109', mode='label',is_training=True,transform=None):
        super().__init__(transform, is_training)
        self.dataset = []
        self.name=[]
        image_path = f'{root}/images'
        mask_path = f'{root}/{mode}'


        for image_name in os.listdir(image_path):
            mask_name = image_name.split('.')[0] + '.json'
            self.dataset.append((f'{image_path}/{image_name}', f'{mask_path}/{mask_name}'))
            self.name.append(image_name)

    @staticmethod
    def parse_txt(mask_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """

        with open(mask_path, 'r') as load_f:
            load_dict = json.load(load_f)

        info = load_dict['shapes']
        polygons = []
        transcripts0 = []
        transcripts1 = []
        for i in range(len(info)):
            points = info[i]['points']
            points = np.array(points).astype(np.int32)
            label = info[i]['label']
            if label == "poi_val0":
                val = info[i]['description']
                transcripts0.append(val)
            polygons.append(TextInstance(points, 'c', label))
            if label == "poi_val1":
                val = info[i]['description']
                transcripts1.append(val)
            polygons.append(TextInstance(points, 'c', label))
        transcripts = transcripts0 + transcripts1

        # print("Loaded transcripts:", transcripts)  # 添加打印语句
        return polygons, transcripts


    def __getitem__(self, item):
        image_path, mask_path = self.dataset[item]
        idx=self.name[item]

        # Read image data
        image = pil_load_img(image_path)
        # print('imagen',image)

        try:
            polygons,transcripts = self.parse_txt(mask_path)
        except:
            polygons = None

        # print("poltygons",polygons)

        return self.get_training_data(image, polygons,transcripts,image_id=idx, image_path=image_path)


    def __len__(self):
        return len(self.dataset)


