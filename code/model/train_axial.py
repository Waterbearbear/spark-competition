# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/24 15:21
# @Author  : Xin Zhang
# @FileName: train.py
# @Software: PyCharm
import torch
import argparse
import torch.utils.data
from model import model_axial
from datasets import dataset , transform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from torch.utils.tensorboard import SummaryWriter
import config

from sklearn.metrics import classification_report , confusion_matrix , precision_score
from utils import util


def train(opt, train_dataloader, val_dataloader,model):
    # writer = SummaryWriter('./logs')
    iter = 0
    f1_max = 0

    print("Start Training")

    for epoch in range(opt.epochs):
        if torch.cuda.is_available():
            model.net.cuda()
        loss = 0
        count = 0
        for i, data in enumerate(train_dataloader):


            count += 1

            # 最后一轮batchsize不一定为opt.batchsize
            batch_size = data[0].shape[0]
            model.set_input(data[0],data[1])

            model.forward()
            model.optimizer.zero_grad()


            # label = transform.onehot(batch_size,opt.num_classes,model.label)
            # label = transform.onehot(batch_size,opt.num_classes,model.label).long()

            print("train label:",model.label)

            print("model predi: ",model.pre)
            # probs = model.probability(model.pre)
            model.label = model.label.long()

            model.loss = model.loss_fn(model.pre, model.label)

            model.loss.backward()


            model.optimizer.step()
            loss = loss + model.loss

            model.update_learning_rate()

            if i % 10 == 0:
                print("epoch:{}, iter:{}, loss:{}".format(epoch, i, model.loss))
                # target_img, pre_img = model.getPre()
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
        # model.save_network(model.net, epoch, opt.gpu_ids)
        print("epoch:{}, mean_loss:{}".format(epoch, mean_loss))
        if epoch%config.eval_epochs == 0:


            acc, f1, prec, rec = validate(config,val_dataloader,model)
            if f1 > f1_max:
                f1_max = f1
                model.save_network(model.net,epoch,model.gpu_ids,f1_max)


def validate(opt, dataloader, model):

    print("Validation started...")
    gts,predictions = [],[]

    model.net.eval()


    for i, data in enumerate(dataloader):

        data[0],data[1] = data[0].cuda(),data[1].cuda()

        with torch.no_grad():
            model.set_input(data[0], data[1])

            model.forward()


            probs = model.probability(model.pre)

            preds = torch.argmax(probs,dim = 1).cpu().numpy()

        label = data[1].cpu().detach().numpy()

        predictions.extend(preds)
        gts.extend(label)

    predictions = np.array(predictions, dtype=np.int32)
    gts = np.array(gts, dtype=np.int32)

    print("predictions: ",predictions)
    print("gts: ",gts)

    acc, f1, prec, rec = util.clf_metrics(predictions=predictions,
                                          targets=gts,
                                          average="macro")

    report = classification_report(gts, predictions, output_dict=True)

    print("VALIDATION | Accuracy {:.4f} | F1 {:.4f} | Precision {:.4f} | "
             "Recall {:.4f}".format(acc, f1, prec, rec))

    print("Validation end")


    return acc, f1, prec, rec
        # model.optimize_parameters()





    # writer.add_scalar('epoch', mean_loss, epoch)
    # model.save_network(model.net, epoch, opt.gpu_ids)



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train_path', type=str, default=r'F:\DATA\Lumbar\lumbar_train150\lumbar_train150')
    # parser.add_argument('--train_json', type=str, default=r'F:\DATA\Lumbar\lumbar_train150_annotation.json')
    # parser.add_argument('--checkpoints_dir', type=str, default='./checkpoint')
    # parser.add_argument('--isTrain', type=bool, default=True)
    # parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
    # parser.add_argument('--epoch', type=int, default=250, help='epoch')
    # parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    # parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    # parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    # opt = parser.parse_args()


    train_dataset = dataset.axialdataset(data_root_path = config.trainPath,
                                         data_json_path = config.trainjsonPath,
                                         is_train=True,
                                         transform=transform.train_transforms())

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=config.batch_size,
                                                   shuffle=True)

    val_dataset = dataset.axialdataset(data_root_path = config.valPath,
                                         data_json_path = config.valjsonPath,
                                         is_train=True,
                                         transform=transform.val_transforms())


    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=config.batch_size,
                                                 shuffle=False)



    model = model_axial.Model(config)
    train(config, train_dataloader,val_dataloader, model)

    # validate(config,val_dataloader,model)
