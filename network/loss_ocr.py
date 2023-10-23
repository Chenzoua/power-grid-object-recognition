import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from torch.nn import CTCLoss

class OCRLoss(nn.Module):

    def __init__(self):
        super(OCRLoss, self).__init__()
        self.ctc_loss = CTCLoss(zero_infinity=True)  # pred, pred_len, labels, labels_len

    def forward(self, *inputs):
        gt, pred = inputs[0], inputs[1]
        print('gt',gt)
        print('pred',pred)
        loss = self.ctc_loss(pred[0], gt[0], pred[1], gt[1])
        return loss


class TextLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.recogitionLoss = OCRLoss()


    def forward(self, inputs, y_true_recog, y_pred_recog):


        # pointer_pred = inputs[:, 0]
        # dail_pred = inputs[:, 1]
        text_pred=inputs[:,0]
        print("inputs",inputs.shape)
        print("text_pred",text_pred.shape)

        # modify tr_loss cross_entropy loss to dice loss

        # loss_pointer = self.dice_loss(pointer_pred, pointer_mask, train_mask)
        # loss_dail = self.dice_loss(dail_pred, dail_mask, train_mask)
        # loss_text = self.dice_loss(text_pred, text_mask, train_mask)

        #
        # loss_pointer=loss_pointer.mean()
        # loss_dail=loss_dail.mean()
        # loss_text=loss_text.mean()

        recognition_loss = self.recogitionLoss(y_true_recog, y_pred_recog)


        return recognition_loss

