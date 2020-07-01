#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/24 15:21
# @Author  : Xin Zhang
# @FileName: train.py
# @Software: PyCharm
import torch
import argparse
import torch.utils.data
from model import Model
from dataset import sparkset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter


def train(opt, dataloader, model):
    # writer = SummaryWriter('./logs')
    iter = 0
    for epoch in range(opt.epoch):
        loss = 0
        count = 0
        for i, data in enumerate(dataloader):
            count += 1
            model.set_input(data[0], data[1], data[2])
            model.optimize_parameters()
            loss += model.loss
            if i % 10 == 0:
                print("epoch:{}, iter:{}, loss:{}".format(epoch, i, model.loss))
                target_img, pre_img = model.getPre()
                target_img = np.array(target_img[0, :, :, :])
                pre_img = np.array(pre_img[0, :, :, :])
                fig = plt.figure()
                sns_plot = sns.heatmap(target_img.sum(axis=0))
                fig.savefig('./logs/img/' + str(epoch) + '_' + str(i) + '_1.png', bbox_inches='tight')
                plt.close()
                fig = plt.figure()
                sns_plot = sns.heatmap(pre_img.sum(axis=0))
                fig.savefig('./logs/img/' + str(epoch) + '_' + str(i) + '_2.png', bbox_inches='tight')
                plt.close()
        mean_loss = loss / count
        # writer.add_scalar('epoch', mean_loss, epoch)
        # model.save_network(model.net, epoch, opt.gpu_ids)
        print("epoch:{}, mean_loss:{}".format(epoch, mean_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default=r'F:\DATA\Lumbar\lumbar_train150\lumbar_train150')
    parser.add_argument('--train_json', type=str, default=r'F:\DATA\Lumbar\lumbar_train150_annotation.json')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoint')
    parser.add_argument('--isTrain', type=bool, default=True)
    parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
    parser.add_argument('--epoch', type=int, default=250, help='epoch')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    opt = parser.parse_args()
    dataset = sparkset(opt.train_path, opt.train_json)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=opt.batchsize, shuffle=True)
    model = Model(opt)
    train(opt, dataloader, model)
