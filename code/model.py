#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/25 12:12
# @Author  : Xin Zhang
# @FileName: model.py
# @Software: PyCharm
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from net import unet


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Model:
    def name(self):
        return 'Model'

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir)
        self.inputImg = self.Tensor(opt.batchsize, 1, 256, 256)
        self.target = self.Tensor(opt.batchsize, 1, 64, 64)
        self.preCoord = []
        self.net = unet().cuda()
        self.loss_fn = nn.MSELoss()
        if self.isTrain:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def set_input(self, img, target, meta):
        self.inputImg.resize_(img.size()).copy_(img)
        self.target.resize_(target.size()).copy_(target)

    def forward(self):
        self.img = Variable(self.inputImg)
        self.pre = self.net(self.img)

    def backward(self):
        self.loss = self.loss_fn(self.target, self.pre)
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def getPre(self):
        return self.target.cpu().detach().numpy(), self.pre.cpu().detach().numpy()

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, epoch_label, gpu_ids):
        save_filename = '%s.pth' % epoch_label
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=0)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        pass
