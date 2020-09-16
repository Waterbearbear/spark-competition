import os
import torch
import numpy as np
from copy import deepcopy
import torch.nn as nn
from torch.utils import mkldnn as mkldnn_utils
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from ..backbone.net import NestedUNet, UNet_3Plus


class CoordModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.loss_fn = nn.MSELoss()
        self.backbone = model

    def _train(self, img, target):
        img = img.type(torch.FloatTensor)
        pre = self.backbone(img)
        loss = self.loss_fn(target, pre)
        return loss

    def _inference(self, img, meta):
        pre = self.backbone(img).detach().numpy()
        heatlist = pre[0, :, :, :]
        ori_size = meta['ori']
        prePts = []
        for i in range(11):
            index = np.where(heatlist[i] == np.max(heatlist[i]))
            prePts.append((index[1].tolist()[0], index[0].tolist()[0]))
        prePts = np.array(prePts)
        preCoord = prePts.copy() * (float(ori_size) / float(256), float(ori_size) / float(256))
        return prePts, preCoord


    def forward(self, args):
        if self.training:
            return self._train(*args)
        else:
            return self._inference(*args)
