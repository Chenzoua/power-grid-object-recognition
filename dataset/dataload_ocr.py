import copy
import cv2
import torch
import numpy as np
from PIL import Image


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


class TextInstance(object):
    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text
        self.bottoms = None
        self.e1 = None
        self.e2 = None

        if self.text != "#":
            self.label = 1
        else:
            self.label = -1

        self.points = np.array(points)


    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(object):

    def __init__(self, transform, is_training=False):
        super().__init__()
        self.transform = transform
        self.is_training = is_training



    @staticmethod
    def make_text_region(polygon,mask):


        cv2.fillPoly(mask, [polygon.points.astype(np.int32)], 1)    #make text_region

        return mask



    def get_training_data(self, image, polygons,transcripts, image_id, image_path):

        H, W, _ = image.shape
        bboxs = []
        bboxs0 = []
        bboxs1 = []

        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))

        pointer_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        dail_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        text_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)

        if polygons is not None:
            for i, polygon in enumerate(polygons):
                if polygon.text == 'poi':
                    pointer_mask = self.make_text_region(polygon, pointer_mask)

                if polygon.text == 'sca0' or polygon.text == 'sca1':
                    dail_mask = self.make_text_region(polygon, dail_mask)

                # if polygon.text == 'poi_val0':
                # # if polygon.text != 'poi' and polygon.text != 'sca0' and polygon.text != 'sca1':
                #     text_mask = self.make_text_region(polygon, text_mask)
                #     bboxs0 = polygon.points.reshape((1, 8))
                #     # print('bboxs0',bboxs0)
                #     # print(bboxs0.shape)

                if polygon.text == 'poi_val1':
                # if polygon.text != 'poi' and polygon.text != 'sca0' and polygon.text != 'sca1':
                    text_mask = self.make_text_region(polygon, text_mask)
                    bboxs1 = polygon.points.reshape((1, 8))
                    # print('bboxs1',bboxs1)
                    # print(bboxs1.shape)
            # bboxs = np.concatenate((bboxs0, bboxs1), axis=0)
            # print('bboxs', bboxs)
            # print(bboxs.shape)


        train_mask = np.ones(image.shape[:2], np.uint8)
        # # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        if not self.is_training:    #test condition

            meta = {
                'image_id': image_id,
                'Height': H,
                'Width': W,
                'trans':transcripts
            }

            return image, pointer_mask,dail_mask,text_mask,train_mask, meta




        return image,  text_mask, train_mask, bboxs1, transcripts


    def __len__(self):
        raise NotImplementedError()
