import torch
import argparse
import torch.utils.data
from ..model.coordmodel import Model
from ..datasets.dataset import CoordDataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.util import compute_nme
from torch.utils.tensorboard import SummaryWriter


def train(opt, train_dataloader, val_dataloader, model):
    # writer = SummaryWriter('./logs')
    iter = 0
    val_dst_min = 1000
    for epoch in range(opt.epoch):
        loss = 0
        count = 0
        dst = 0
        nme_sum = 0
        for i, data in enumerate(train_dataloader):
            count += 1
            model.set_input(data[0], data[1], data[2])
            ori_size = data[2]['ori']
            pts = data[2]['pts'].numpy()[0]
            ps = data[2]['ps'].numpy()[0][0]
            model.optimize_parameters()
            loss += model.loss
            model.getHeatmap()
            prepts, precoord = model.getPreCoord()
            tpts = data[2]['tpts'].numpy()[0]
            nme_batch = compute_nme(prepts, pts)
            nme_sum += nme_batch
            dst += np.array(np.mean(np.sqrt(np.sum(np.square(prepts - tpts), axis=1)))) * \
                   [float(ori_size) / float(256)] * ps

        target_img, pre_img = model.getHeatmap()
        dst = dst / count
        nme = nme_sum / count
        # print(nme)
        # print(tpts)
        # print(dst)
        # print(precoord)
        # target_img = np.array(target_img[0, :, :, :])
        # pre_img = np.array(pre_img[0, :, :, :])
        # fig = plt.figure()
        # sns_plot = sns.heatmap(target_img.sum(axis=0))
        # fig.savefig('./logs/img/' + str(epoch) + '_' + str(i) + '_1.png', bbox_inches='tight')
        # plt.close()
        # fig = plt.figure()
        # sns_plot = sns.heatmap(pre_img.sum(axis=0))
        # fig.savefig('./logs/img/' + str(epoch) + '_' + str(i) + '_2.png', bbox_inches='tight')
        # plt.close()
        mean_loss = loss / count
        # writer.add_scalar('epoch', mean_loss, epoch)
        print("epoch:{}, mean_loss:{}, lr:{}".format(epoch, mean_loss, model.scheduler.get_lr()))
        val_dst = validate(epoch, val_dataloader, model)
        if val_dst < val_dst_min:
            val_dst_min = val_dst
            model.save_network(model.net, 0, opt.gpu_ids)
            print('val_dst update')
        model.update_learning_rate()


def validate(epoch, dataloader, model):
    count = 0
    nme_sum = 0
    dst = 0
    for i, data in enumerate(dataloader):
        count += 1
        model.set_input(data[0], data[1], data[2])
        model.test()
        ori_size = data[2]['ori']
        pts = data[2]['pts'].numpy()[0]
        ps = data[2]['ps'].numpy()[0][0]
        model.getHeatmap()
        prepts, precoord = model.getPreCoord()
        tpts = data[2]['tpts'].numpy()[0]
        nme_batch = compute_nme(prepts, pts)
        nme_sum += nme_batch
        dst += np.array(np.mean(np.sqrt(np.sum(np.square(prepts - tpts), axis=1)))) * \
               [float(ori_size) / float(256)] * ps
    target_img, pre_img = model.getHeatmap()
    dst = dst / count
    nme = nme_sum / count
    print("val dst:{}, nme:{}".format(dst, nme))
    # target_img = np.array(target_img[0, :, :, :])
    # pre_img = np.array(pre_img[0, :, :, :])
    # fig = plt.figure()
    # sns_plot = sns.heatmap(target_img.sum(axis=0))
    # fig.savefig('./logs/img/' + str(epoch) + '_' + str(i) + '_3.png', bbox_inches='tight')
    # plt.close()
    # fig = plt.figure()
    # sns_plot = sns.heatmap(pre_img.sum(axis=0))
    # fig.savefig('./logs/img/' + str(epoch) + '_' + str(i) + '_4.png', bbox_inches='tight')
    # plt.close()
    return dst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default=r'F:\DATA\Lumbar\train')
    parser.add_argument('--train_json', type=str, default=r'F:\DATA\Lumbar\train.json')
    parser.add_argument('--val_path', type=str, default=r'F:\DATA\Lumbar\lumbar_train51\train')
    parser.add_argument('--val_json', type=str, default=r'F:\DATA\Lumbar\lumbar_train51_annotation.json')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoint/1')
    parser.add_argument('--isTrain', type=bool, default=True)
    parser.add_argument('--continue_train', type=bool, default=False)
    parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    opt = parser.parse_args()
    train_dataset = CoordDataset(opt.train_path, opt.train_json, is_flip=True, is_rot=True, is_train=True)
    val_dataset = CoordDataset(opt.val_path, opt.val_json, is_flip=False, is_rot=False, is_train=False)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batchsize, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=opt.batchsize, shuffle=True)
    model = Model(opt)
    train(opt, train_dataloader, val_dataloader, model)
