# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/24 15:21
# @Author  : Xin Zhang
# @FileName: train.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
import pytorch_warmup as warmup
import argparse
import torch.utils.data
from model import model_axial
from torchsampler import ImbalancedDatasetSampler
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
            model.optimize_parameters()

            # model.forward()
            # model.optimizer.zero_grad()
            #
            #
            # # label = transform.onehot(batch_size,opt.num_classes,model.label)
            # # label = transform.onehot(batch_size,opt.num_classes,model.label).long()
            #
            # # print("train label:",model.label)
            #
            # # print("model predi: ",model.proconfig.step_print)
            # # proconfig.step_print = model.probability(model.pre)
            # model.label = model.label.long()
            #
            # model.loss = model.loss_fn(model.pre, model.label)
            #
            # model.loss.backward()
            #
            #
            # model.optimizer.step()
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


def validate(dataloader):

    print("Validation started...")
    gts,predictions = [],[]

    model.eval()


    for i, data in enumerate(dataloader):

        data[0],data[1] = data[0].cuda(),data[1].cuda()

        # print("data[0]: ",data[0])
        # print("data[1]: ",data[1])

        with torch.no_grad():
            model.set_input(data[0], data[1])

            model.forward()


            preds = torch.argmax(model.proconfig.step_print,dim = 1).cpu().numpy()


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

def val():
        model.eval()
        test_loss = 0
        correct = 0
        results = []

        TP = 0
        TN = 0
        FN = 0
        FP = 0

        # criteria = nn.CrossEntropyLoss()
        criteria = model_axial.focal_loss(alpha=[0.1,0.2,0.2,0.5,0.5], gamma=2, num_classes = 5)
        # Don't update model
        with torch.no_grad():
            tpr_list = []
            fpr_list = []

            predlist = []
            scorelist = []
            targetlist = []
            # Predict
            for batch_index, (img,label) in enumerate(val_dataloader):
                data, target = img.to(device), label.to(device)

                #            data = data[:, 0, :, :]
                #            data = data[:, None, :, :]
                output = model(data)

                test_loss += criteria(output, target.long())
                score = F.softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)
                #             print('target',target.long()[:, 2].view_as(pred))
                # correct += pred.eq(target.long().view_as(pred)).sum().item()

                #             print(output[:,1].cpu().numpy())
                #             print((output[:,1]+output[:,0]).cpu().numpy())
                #             predcpu=(output[:,1].cpu().numpy())/((output[:,1]+output[:,0]).cpu().numpy())
                targetcpu = target.long().cpu().numpy()

                predlist = np.append(predlist, pred.cpu().numpy())
                scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
                targetlist = np.append(targetlist, targetcpu)

        acc, f1, prec, rec = util.clf_metrics(predictions=predlist,
                                              targets=targetlist,
                                              average="macro")

        report = classification_report(targetlist, predlist, output_dict=True)

        print("VALIDATION | Accuracy {:.4f} | F1 {:.4f} | Precision {:.4f} | "
              "Recall {:.4f}".format(acc, f1, prec, rec))

        print(report)

        print("Validation end")


        return acc, f1, prec, rec

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
                                                   sampler=ImbalancedDatasetSampler(train_dataset,callback_get_label = callback_get_label),
                                                   num_workers= config.n_threads,
                                                   shuffle=False)
    
    
    val_dataset = dataset.axialdataset(data_root_path = config.valPath,
                                         data_json_path = config.valjsonPath,
                                         is_train=True,
                                         transform=None)


    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=config.batch_size,
                                                 shuffle=False)

    model = models.resnet152(pretrained=not config.pretrained).cuda()
    model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.classifier = nn.Linear(in_features=1664, out_features=5, bias=True)

    modelname = config.model_name

    if config.pretrained == True:
        print("loading pretrain model parameters")

        pretrain_dict = torch.load(config.pretrainedPath)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)

    # model = models.ResNet152(pretrained=True, progress=True)

    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #
    # model.fc = nn.Linear(in_features=2048, out_features=5, bias=True)

    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=50)


    AUC_best = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f1_max = 0

    for epoch in range(1,config.epochs + 1):

        model.train()

        train_loss = 0
        train_correct = 0

        for batch_index, (img,label) in enumerate(train_dataloader):

            if batch_index%20 == 0:
                util.imshow(img[0])

            # move data to device
            # print("type target: ",type(target))
            img, target = img.to(device), label.to(device)

            # print("img: ",img)
            # print("target.shape: ",target.shape)
            # print(target)
            #        data = data[:, 0, :, :]
            #        data = data[:, None, :, :]

            # data, targets_a, targets_b, lam = mixup_data(data, target.long(), alpha, use_cuda=True)

            optimizer.zero_grad()
            output = model(img)
            # print(output.shape)
            # print("output: ",output)
            # [ 0.2019,  0.2341],
            #         [ 0.1498,  0.0449],
            #         [ 0.0766, -0.1680],
            #         [-0.1270,  0.1958],
            #         [-0.2086,  0.4554],

            # criteria = nn.CrossEntropyLoss()
            criteria = model_axial.focal_loss(alpha=[0.1,0.2,0.2,0.5,0.5], gamma=2, num_classes = 5)
            loss = criteria(output, target.long())
            # loss = lam * criteria(output, targets_a) + (1 - lam) * criteria(output, targets_b)
            train_loss =  train_loss + loss.item()

            # 学习率进行warm-up
            # 和使用余弦退火学习率衰减

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            warmup_scheduler.dampen()
            # print(optimizer.state_dict()['param_groups'][0]['lr'])
            pred = output.argmax(dim=1, keepdim=True)

            train_correct += pred.eq(target.long().view_as(pred)).sum().item()

            # Display progress and write to tensorboard
            if batch_index % config.step_print == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                    epoch, batch_index, len(train_dataloader),
                    100.0 * batch_index / len(train_dataloader), loss.item() / config.step_print))

        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss / len(train_dataloader.dataset), train_correct, len(train_dataloader.dataset),
            100.0 * train_correct / len(train_dataloader.dataset)))
        f = open('E:\BME\competition\spark\code\checkpoint\{}.txt'.format(modelname), 'a+')
        f.write('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss / len(train_dataloader.dataset), train_correct, len(train_dataloader.dataset),
            100.0 * train_correct / len(train_dataloader.dataset)))
        f.write('\n')
        f.close()

        if epoch % config.eval_epochs == 0:

            acc, f1, prec, rec = val()
            print("F1: ",f1)
            if f1 > f1_max:
                print("Is best!pre F1:%.4f,best F1:%.4f"%(f1_max,f1))
                f1_max = f1
                torch.save(model.state_dict(), "E:\BME\competition\spark\code\checkpoint\{}_epoch{}_F1_{:.4f}_best.pth".format(modelname,epoch,f1_max))

        if epoch == config.epochs:
            torch.save(model.state_dict(), "E:\BME\competition\spark\code\checkpoint\{}_epoch{}_F1_{:.4f}.pth".format(modelname,epoch,f1_max))


