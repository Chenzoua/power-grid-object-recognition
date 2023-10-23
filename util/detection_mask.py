import numpy as np
import cv2
from util.config import config as cfg
import torch


class TextDetector(object):

    def __init__(self, model):
        self.model = model

        # evaluation mode
        model.eval()

    def detect1(self, image):

        with torch.no_grad():
            # get model output
            pointer_pred,dail_pred,text_pred,pred_recog1,pred_recog2,std_points= self.model.forward_test(image)


        image = image[0].data.cpu().numpy()


        output = {
            'image': image,
            'pointer': pointer_pred,
            'dail': dail_pred,
            'text': text_pred,
            'reco1':pred_recog1,
            'reco2':pred_recog2,
            'std':std_points
        }
        return output

    def detect2(self, image):

        with torch.no_grad():
            # get model output
            text_pred,pred_recog,bboxes= self.model.forward_test2(image)


        image = image[0].data.cpu().numpy()


        output = {
            'image': image,
            'text': text_pred,
            'reco':pred_recog,
            'box': bboxes,
        }
        return output




















