import torch

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from datasets import transform
import torch.nn.functional as F
import numpy as np
from backbone.layers import DoubleNet

import config
# from resnest.torch import resnest101, resnest50

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            # print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            # print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds,
                                  dim=1)  # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


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
        self.gpu_ids = opt.gpu
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir)

        self.inputImg = self.Tensor(opt.batch_size, 1, 256, 256)
        # self.label = self.Tensor(opt.batch_size, 1, 64, 64)

        self.label = self.Tensor(opt.batch_size, opt.num_classes)

        self.preCoord = []

        self.net = models.densenet169(pretrained=True)
        self.net.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.classifier = nn.Linear(in_features=1664, out_features=5, bias=True)

        self.net = self.net if self.gpu_ids else self.net

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        # self.loss_fn = FocalLoss()
        self.loss_fn = nn.CrossEntropyLoss()

        if self.isTrain:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

    def set_input(self, img, label):
        self.inputImg.resize_(img.size()).copy_(img)
        self.label.resize_(label.size()).copy_(label)
        if self.gpu_ids:
            self.inputImg = self.inputImg
            self.label = self.label.long()

    def forward(self):
        self.img = Variable(self.inputImg)
        self.pre = self.net(self.img)
        self.probs = self.sigmoid(self.pre)

        # self.probs = self.softmax(self.pre)

    def backward(self):

        self.loss = self.loss_fn(self.pre, self.label)
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def getPre(self):
        return self.label.cpu().detach(), self.pre.cpu().detach()

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
    def save_network(self, network, epoch_label, gpu_ids, best_f1):
        save_filename = 'DenseNet_epoch%s_F1_%f.pth' % (epoch_label, best_f1)
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(network.cpu().state_dict(), save_path)
        # if len(gpu_ids) and torch.cuda.is_available():
        #     network.cuda(device=0)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, F1score):
        save_filename = '%s_epoch%d_F1%f.pth' % (network_label, epoch_label, F1score)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        # self.scheduler.step()
        pass


class Bagging_Model(nn.Module):
    """
    Attention Block
    """
    def __init__(self,model_name,net_num,num_classes,part,nets_path = None):
        super(Bagging_Model, self).__init__()

        self.model_name = model_name
        self.net_num = net_num
        self.num_classes = num_classes
        self.part = part

        if self.model_name == 'ResNet50':
            net = models.resnet50(pretrained=False)
            net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            net.fc = nn.Linear(in_features=2048, out_features= self.num_classes , bias=True)

        # elif self.model_name == "ResNeSt101":
        #     net = resnest101(pretrained=not config.pretrained)
        #     # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #     net.fc = nn.Linear(in_features=2048, out_features= self.num_classes , bias=True)
        #
        # elif self.model_name == "ResNeSt50":
        #     net = resnest50(pretrained=not config.pretrained)
        #     # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #     net.fc = nn.Linear(in_features=2048, out_features= self.num_classes , bias=True)

        elif self.model_name == 'MobileNet_v2':
            net = models.mobilenet_v2(pretrained= False)

            net.classifier[1] = nn.Linear(in_features=1280, out_features=self.num_classes)

        else:
            raise ValueError("select a model name : ",self.model_name)

        # if config.gpu == True:
        #     net = net

        self.nets_path = nets_path
        ####### 按不同F1分数进行bagging ##########

        # self.bagging_net = {"netv%d" % i: net for i in range(1, self.net_num + 1)}
        # for key in config.target_names:
        #
        #     parameter_path = os.path.join(self.nets_path,self.model_name + "_%sbest.pth"%key)
        #     print(parameter_path)
        #
        #     pretrained_dict = torch.load(parameter_path)
        #     model_dict = self.bagging_net["net%s"%key].state_dict()
        #
        #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #     model_dict.update(pretrained_dict)
        #
        #     self.bagging_net["net%s"%key].load_state_dict(model_dict)


        ####### 按不同F1分数进行bagging ##########


        ##### 按K折交叉验证进行bagging##############
        self.bagging_net = {'net%d' % i:net for i in range(config.k_fold)}

        for fold in range(config.k_fold):

            parameter_path = os.path.join(self.nets_path, self.model_name + "_%s_fold%d_best.pth" % (self.part, fold))
            # print(parameter_path)

            pretrained_dict = torch.load(parameter_path)
            model_dict = self.bagging_net["net%s"%fold].state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)

            self.bagging_net["net%s"%fold].load_state_dict(model_dict)

        ##### 按K折交叉验证进行bagging##############

    def forward(self,img):

        ##############################    K_fold bagging    ##############################
        output = 0

        with torch.no_grad():
            for fold in range(config.k_fold):

                # print(key)
                output = torch.add(self.bagging_net["net%d"%fold](img),output)

            output.div_(config.k_fold)

        ##############################    K_fold bagging    ##############################

        return output



class Bagging_Double_Model(nn.Module):
    def __init__(self, model_name, net_num, num_classes, part, nets_path=None):
        super(Bagging_Double_Model, self).__init__()

        self.model_name = model_name
        self.net_num = net_num
        self.num_classes = num_classes
        self.part = part
        self.nets_path = nets_path

        # if self.model_name == 'ResNet18':
        #     if self.part == "vertebra":
        #         model = DoubleNet(modelname=self.model_name, num_classes=self.num_classes,
        #                           pretrained_path=None)
        #     elif self.part == "disc":
        #         model = DoubleNet(modelname=self.model_name, num_classes=self.num_classes,
        #                           pretrained_path=None)
        #     else:
        #         print("part error")
        #         raise ValueError
        # else:
        #     print("didn't chose a net")
        #     raise  ValueError

        if self.model_name == "ResNet18":
            pass
        elif self.model_name == "ResNeSt18":
            pass
        elif self.model_name == "MobileNet_v2":
            pass
        else:
            print("didn't choose a net")
            raise ValueError

        model = DoubleNet(modelname = self.model_name,
                          phase = self.part,
                          pretrained_path=None,
                          bf16=False)

        # if self.model_name == 'ResNet50':
        #     net = models.resnet50(pretrained=False)
        #     net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #     net.fc = nn.Linear(in_features=2048, out_features= self.num_classes , bias=True)
        #
        # elif self.model_name == "ResNeSt101":
        #     net = resnest101(pretrained=not config.pretrained)
        #     # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #     net.fc = nn.Linear(in_features=2048, out_features= self.num_classes , bias=True)
        #
        # elif self.model_name == "ResNeSt50":
        #     net = resnest50(pretrained=not config.pretrained)
        #     # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #     net.fc = nn.Linear(in_features=2048, out_features= self.num_classes , bias=True)

        # net = models.resnet152(pretrained=False)
        # net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # net.fc = nn.Linear(in_features=2048, out_features= self.num_classes , bias=True)

        # if config.gpu == True:
        #     net = net

        ####### 按不同F1分数进行bagging ##########

        # self.bagging_net = {"netv%d" % i: net for i in range(1, self.net_num + 1)}
        # for key in config.target_names:
        #
        #     parameter_path = os.path.join(self.nets_path,self.model_name + "_%sbest.pth"%key)
        #     print(parameter_path)
        #
        #     pretrained_dict = torch.load(parameter_path)
        #     model_dict = self.bagging_net["net%s"%key].state_dict()
        #
        #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #     model_dict.update(pretrained_dict)
        #
        #     self.bagging_net["net%s"%key].load_state_dict(model_dict)

        ####### 按不同F1分数进行bagging ##########

        ##### 按K折交叉验证进行bagging##############
        self.bagging_net = {'net%d' % i: model for i in range(config.k_fold)}

        for fold in range(config.k_fold):
            parameter_path = os.path.join(self.nets_path, "%s_%s_fold%d_best.pth" % (self.model_name, self.part, fold))
            # print(parameter_path)

            pretrained_dict = torch.load(parameter_path)
            model_dict = self.bagging_net["net%s" % fold].state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)

            self.bagging_net["net%s" % fold].load_state_dict(model_dict)

        ##### 按K折交叉验证进行bagging##############

    def forward(self, sag_img, axial_img):

        # print("self.baggingnet: ",self.bagging_net)

        # bagging_output = torch.zeros([self.net_num,config.num_classes])

        ########  F1 score bagging ############
        # output = {}
        # with torch.no_grad():
        #     for i,key in enumerate(config.target_names):
        #
        #         # print(key)
        #         output = torch.add(self.bagging_net["net"%key](img),output)
        #
        #     output.div_(config.k_fold)

        ########  F1 score bagging ############

        ##############################    K_fold bagging    ##############################
        output = 0

        with torch.no_grad():
            for fold in range(config.k_fold):
                # print(key)
                output = torch.add(self.bagging_net["net%d" % fold](sag_img, axial_img), output)

            output.div_(config.k_fold)

        ##############################    K_fold bagging    ##############################

        #
        # pred = output.argmax(dim=1, keepdim=True)

        # print("bagging_output:",bagging_output)
        # print(type(bagging_output))
        # print("bagging.shape: ",bagging_output.shape)

        return output
