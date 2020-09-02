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
import os
# import model_axial
from model import model_axial

import numpy as np
import seaborn as sns
import config
import time



from sklearn.metrics import classification_report , confusion_matrix , precision_score
from utils import util
from torchsampler import ImbalancedDatasetSampler
from resnest.torch import resnest50,resnest101
from tensorboardX import SummaryWriter
from backbone.layers import DoubleNet

from datasets import dataset,transform


def val(model,weights,dataloader,device):
        model.eval()
        test_loss = 0
        correct = 0
        results = []

        TP = 0
        TN = 0
        FN = 0
        FP = 0

        criteria = nn.CrossEntropyLoss(weight = weights)
        # criteria = model_axial.focal_loss(alpha=weights, gamma=2, num_classes = num_classes)
        # Don't update model
        with torch.no_grad():
            tpr_list = []
            fpr_list = []

            predlist = []
            scorelist = []
            targetlist = []
            # Predict

            for batch_index, (axial_img,sag_img,target,disc_path,identification,studyID) in enumerate(dataloader):
                # Axial和Sag双路
                # axial_img, sag_img,target = axial_img.to(device),sag_img.to(device), label.to(device)
                # output = model(sag_img,axial_img)
                # 仅用sag
                sag_img = sag_img.repeat([1,3,1,1])

                output = model(sag_img)

                # print("output: ",output)
                # print("output.shape",output.shape)

                test_loss += criteria(output, target.long())
                score = F.softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)

                # print("pred",pred)
                # print("pred.shape",pred.shape)



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

        f1_dict = {}
        #
        # map_dict = {'v1':'0.0',
        #             'v2':'1.0',
        #             'v3':'2.0',
        #             'v4':'3.0',
        #             'v5':'4.0'
        # }
        # for key in config.target_names:
        #     items = report[map_dict[key]]
        #
        #     print(items)
        #     f1_dict[key] = items['f1-score']

        print("VALIDATION | Accuracy {:.4f} | F1 {:.4f} | Precision {:.4f} | "
              "Recall {:.4f} | Test loss{:.4f}".format(acc, f1, prec, rec,test_loss))


        print("Validation end")


        return acc, f1, prec, rec,f1_dict,test_loss

    # writer.add_scalar('epoch', mean_loss, epoch)
    # model.save_network(model.net, epoch, opt.gpu_ids)


def callback_get_label(dataset_,idx):

    axial_img,sag_img,label,disc_path,identification,studyID = dataset_[idx]

    return label


# if __name__ == '__main__':
def train_axial_model():

    torch.backends.cudnn.benchmark = True

    localtime = time.asctime(time.localtime(time.time()))

    writer = SummaryWriter(config.checkpoints_dir + '\log')

    file_path = os.path.dirname(__file__)

    # print("file_path:",file_path)
    parent_path = os.path.dirname(file_path)

    # if config.isTrain == 'all':
    # train_all_dataset = dataset_.axialdataset(data_root_path=config.train_allPath,
    #                                          data_json_path=config.train_alljsonPath,
    #                                          is_train='all',
    #                                          transform=transform.train_transforms())

    #### Sag图加 Axial图 数据 #########
    # disc_all_dataset = dataset.SagAxialDataset(data_root_path=config.train_allPath,
    #                                              data_json_path=config.train_alljsonPath,
    #                                              part = 'disc',
    #                                              is_train='all',
    #                                              transform=transform.train_transforms())
    #
    # vertebra_all_dataset = dataset.SagAxialDataset(data_root_path=config.train_allPath,
    #                                            data_json_path=config.train_alljsonPath,
    #                                            part='vertebra',
    #                                            is_train='all',
    #                                            transform=transform.train_transforms())


    ###### 仅仅Sag图  ##########
    disc_all_dataset = dataset.CropSagDataset('disc',is_train=True, transform= transform.train_transforms())
    vertebra_all_dataset = dataset.CropSagDataset('vertebra',is_train = True,transform = transform.train_transforms())

    for parse in ["vertebra","disc"]:
    # for parse in ["disc"]:
        if parse == 'vertebra':
            dataset_ = vertebra_all_dataset
            weights = torch.tensor(config.vertebra_weights,dtype= torch.float )
            num_classes = config.num_classes[0]
        elif parse == 'disc':
            dataset_ = disc_all_dataset
            weights = torch.tensor(config.disc_weights , dtype = torch.float)
            num_classes = config.num_classes[1]
        else:
            print("parse error")
            raise ValueError

        for i in range(config.k_fold):
            if i > 0:
                break
            print(parse)
            modelname = config.model_name
            print(modelname)

            if modelname == "ResNeSt101":
                model = resnest101(pretrained=not config.pretrained)
                # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                model.fc = nn.Linear(in_features=2048, out_features=config.num_classes, bias=True)

            elif modelname == "ResNeSt50":
                model = resnest50(pretrained=not config.pretrained)
                # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                model.fc = nn.Linear(in_features=2048, out_features=config.num_classes, bias=True)

            elif modelname == "ResNet18":
                if parse == "vertebra":
                    model =  DoubleNet(modelname = modelname, num_classes = config.num_classes[0], pretrained_path= config.pretrainedPath[0])
                else:
                    model = DoubleNet(modelname = modelname, num_classes = config.num_classes[1], pretrained_path= config.pretrainedPath[1])

            elif modelname == 'MobileNet_v2':
                if parse == "vertebra":
                    model = models.mobilenet_v2(pretrained=True)

                    model.classifier[1] = nn.Linear(in_features=1280, out_features=config.num_classes[0])
                else:
                    model = models.mobilenet_v2(pretrained=True)

                    model.classifier[1] = nn.Linear(in_features=1280, out_features=config.num_classes[1])

            else:
                raise NotImplementedError("Chose a exist model: ResNet18 or MobileNet_v2")
                # if parse == "vertebra":
                #     model =  DoubleNet(modelname = modelname, num_classes = config.num_classes[0], pretrained_path= config.pretrainedPath[0])
                # else:
                #     model = DoubleNet(modelname = modelname, num_classes = config.num_classes[1], pretrained_path= config.pretrainedPath[1])

                # print("model name False")
                # raise ValueError

            # print(model.net_sag)

            # print(model.net['sag'])
            # print(model.net['axl'])


            # if config.pretrained == True:
            #     print("loading pretrain model parameters")
            #
            #     pretrain_dict = torch.load(config.pretrainedPath)
            #     model_dict = model.state_dict()
            #
            #     pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
            #     model_dict.update(pretrain_dict)
            #     model.load_state_dict(model_dict)
            # else:
            #
            #     print("Loading ImageNet pretrained parameters")



            # for para in list(model.net_sag.parameters()):
            #     para.requires_grad = True
            #
            # for para in list(model.net_axl.parameters()):
            #     para.requires_grad = True
            #
            # for para in list(model.fc.parameters()):
            #     para.requires_grad = True

            # print(model.parameters())
            # print(model)
            #
            for para in list(model.parameters()):
            #     print(para)
                para.requires_grad = True

            # print("k_fold: ",i)

            all_dataset_len = len(dataset_)

            index = (i) * all_dataset_len/config.k_fold

            all_index = [i for i in range(all_dataset_len)]

            val_index   = [i for i in range(int(index),int(index + all_dataset_len/config.k_fold))]

            # print(val_index)

            train_index = [i for i in all_index if i not in val_index]

            train_dataset = torch.utils.data.Subset(dataset_,train_index)
            val_dataset = torch.utils.data.Subset(dataset_,val_index)


            train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                           batch_size=config.batch_size,
                                                           # sampler=ImbalancedDatasetSampler(train_dataset,
                                                           #                                  callback_get_label=callback_get_label),
                                                           num_workers=config.n_threads,
                                                           shuffle=False,
                                                           pin_memory= True)

            val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                         batch_size=config.batch_size,
                                                         shuffle=False,
                                                         pin_memory = True)


            # model = models.ResNet152(pretrained=True, progress=True)

            # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            #
            # model.fc = nn.Linear(in_features=2048, out_features=5, bias=True)
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device('cpu')


            optimizer = optim.Adam(model.parameters(), lr=config.lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= config.decay_epoch)
            # scheduler = optim.lr_scheduler.ExponentialLR(optimizer = optimizer,gamma = 0.98)
            warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period= config.warmup_period)

            AUC_best = 0


            #
            # f1_max = {"v1":0.0,
            #           "v2":0.0,
            #           "v3":0.0,
            #           "v4":0.0,
            #           "v5":0.0}

            # f1_max = {"v1": 0.4303,
            #           "v2": 0.6389,
            #           "v3": 0.3063,
            #           "v4": 0.0690,
            #           "v5": 0.0714}

            f1_max = 0

            for epoch in range(1,config.epochs + 1):

                # if i == 2 and epoch < 47:
                #     continue

                model.train()

                train_loss = 0
                train_correct = 0


                #Axial和Sag双路的Dataloader
                # for batch_index, (axial_img,sag_img,label,disc_path,identification,studyID) in enumerate(train_dataloader):

                ## 仅用Sag的Dataloader
                for batch_index, (axial_img,sag_img,target,disc_path,identification,studyID) in enumerate(
                        train_dataloader):

                    # print("#################################################################################")

                    #使用ResNeSt时，需要3通道
                    sag_img = sag_img.repeat([1,3,1,1])

                    # Axial和Sag双路的
                    # axial_img, sag_img,target = axial_img.to(device), sag_img.to(device),label.to(device)


                    #仅用Sag图


                    # model.cuda(device)

                    # print("img: ",img)
                    # print("target.shape: ",target.shape)
                    # print(target)
                    #        data = data[:, 0, :, :]
                    #        data = data[:, None, :, :]

                    # data, targets_a, targets_b, lam = mixup_data(data, target.long(), alpha, use_cuda=True)

                    optimizer.zero_grad()
                    # Axial和Sag双路的
                    # output = model(sag_img,axial_img)

                    # 仅Sag一路

                    output = model(sag_img)

                    criteria = nn.CrossEntropyLoss(weight = weights)
                    # criteria = model_axial.focal_loss(alpha=weights, gamma=2, num_classes = num_classes)

                    loss = criteria(output, target.long())
                    # loss = lam * criteria(output, targets_a) + (1 - lam) * criteria(output, targets_b)
                    train_loss =  train_loss + loss.item()

                    # 学习率进行warm-up
                    # 和使用余弦退火学习率衰减

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # print(optimizer.state_dict()['param_groups'][0]['lr'])
                    pred = output.argmax(dim=1, keepdim=True)

                    train_correct += pred.eq(target.long().view_as(pred)).sum().item()

                    # Display progress and write to tensorboard
                    if batch_index % config.step_print == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                            epoch, batch_index, len(train_dataloader),
                            100.0 * batch_index / len(train_dataloader), loss.item() / config.step_print))

                scheduler.step()
                warmup_scheduler.dampen()

                print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
                    train_loss / len(train_dataloader.dataset), train_correct, len(train_dataloader.dataset),
                    100.0 * train_correct / len(train_dataloader.dataset)))
                f = open(config.checkpoints_dir + '\{}.txt'.format(modelname + parse), 'a+')
                f.write('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    train_loss / len(train_dataloader.dataset), train_correct, len(train_dataloader.dataset),
                    100.0 * train_correct / len(train_dataloader.dataset)))
                f.write('\n')
                f.close()


                if epoch % config.eval_epochs == 0:

                    acc, f1, prec, rec,f1_dict,val_loss = val(model = model,
                                                              weights = weights,
                                                              dataloader= val_dataloader,
                                                              device = device)

                    # print("F1: ",f1_dict)
                    if f1 > f1_max:
                        print("%d F1score is best!pre F1:%.4f,best F1:%.4f" % (epoch , f1_max, f1))

                        f1_max = f1
                        f = open(config.checkpoints_dir + r'\{}.txt'.format(modelname + parse + "best_f1_fold%d"%i), 'a+')
                        # f.write('\nepoch:%d best f1{v1:%.4f v2:%.4f v3:%.4f v4:%.4f v5:%.4f} %s is best\n'%(epoch,f1_max['v1'],f1_max['v2'],f1_max['v3'],f1_max['v4'],f1_max['v5'],key))
                        f.write('\nfold:%d  epoch:%d best f1:%.5f is best\n'%(i,epoch,f1))

                        f.close()
                        # for net_name in ["sag","axl","fc"]:
                        #     model_save_path = os.path.join(parent_path,config.checkpoints_dir,"{}_{}_{}_fold{}_best.pth".format(modelname,net_name,parse,i))
                        # # print("model_path:",model_save_path)
                        #     if net_name == 'fc':
                        #         torch.save(model.fc.state_dict(),model_save_path)
                        #     else:
                        #         torch.save(model.net[net_name].state_dict(), model_save_path)

                        model_save_path = os.path.join(parent_path, config.checkpoints_dir,
                                                       "{}_{}_fold{}_best.pth".format(modelname, parse, i,epoch))
                        # print("model_path:",model_save_path)
                        torch.save(model.state_dict(),model_save_path)


                    # for key in config.target_names:
                    #     if f1_dict[key] > f1_max[key]:
                    #         print("%s F1score is best!pre F1:%.4f,best F1:%.4f" % (key ,f1_max[key], f1_dict[key]))
                    #         f1_max[key] = f1_dict[key]
                    #
                    #
                    #         # print("parent_path: ",parent_path)
                    #         f = open(config.checkpoints_dir + r'\{}.txt'.format(modelname + "best_f1"), 'a+')
                    #         f.write('\nepoch:%d best f1{v1:%.4f v2:%.4f v3:%.4f v4:%.4f v5:%.4f} %s is best\n'%(epoch,f1_max['v1'],f1_max['v2'],f1_max['v3'],f1_max['v4'],f1_max['v5'],key))
                    #         f.close()
                    #         model_save_path = os.path.join(parent_path,config.checkpoints_dir,"{}_{}_best.pth".format(modelname,key))
                    #
                    #         # print("model_path:",model_save_path)
                    #         torch.save(model.state_dict(), model_save_path)


                    print("all f1 socre:",f1_max)

                # acc, f1, prec, rec,

                writer.add_scalar('log/model%s fold%d train_loss'%(parse,i), train_loss / len(train_dataloader.dataset), epoch)
                writer.add_scalar('log/model%s fold%d val_loss'%(parse,i), val_loss, epoch)
                writer.add_scalar('log/model%s fold%d acc' %(parse, i), acc, epoch)
                writer.add_scalar('log/model%s fold%d prec' %(parse, i), prec, epoch)
                writer.add_scalar('log/model%s fold%d rec' %(parse, i) , rec, epoch)
                writer.add_scalar('log/model%s fold%d learning_rate' %(parse, i),optimizer.param_groups[0]['lr'],epoch)

                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram(tag, value.data.cpu().numpy(), epoch + 1)
                    writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

                if epoch == config.epochs:
                    # for net_name in ["sag", "axl", "fc"]:
                    #     model_save_path = os.path.join(parent_path, config.checkpoints_dir,
                    #                                    "{}_{}_{}_epoch{}_fold{}_final.pth".format(modelname,net_name ,parse, epoch,i))
                    #     # print("model_path:",model_save_path)
                    #     if net_name == 'fc':
                    #         torch.save(model.fc.state_dict(), model_save_path)
                    #     else:
                    #         torch.save(model.net[net_name].state_dict(), model_save_path)
                    model_save_path = os.path.join(parent_path, config.checkpoints_dir,
                                                   "{}_{}_epoch{}_fold{}_final.pth".format(modelname ,parse, epoch,i))
                    # print("model_path:",model_save_path)
                    torch.save(model.state_dict(), model_save_path)


    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()





