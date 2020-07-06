import torch

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from datasets import transform
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, y_pred, y_true):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """

        y_pred = y_pred + self.elipson

        # print("y_true.shape: ",y_true.shape)
        # print("y_pred.shape: ",y_pred.shape)

        cross_entropy = -y_true * torch.log(y_pred)

        weight = torch.pow(1 - y_pred,self.gamma) * y_true

        focalloss = cross_entropy * weight * self.alpha

        reduce_focalloss = torch.max(focalloss)

        return reduce_focalloss




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

        self.label = self.Tensor(opt.batch_size,opt.num_classes)

        self.preCoord = []


        self.net = models.densenet169(pretrained = True).cuda()
        self.net.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.classifier = nn.Linear(in_features= 1664,out_features= 5,bias = True)

        self.net = self.net.cuda() if self.gpu_ids else self.net

        self.sigmoid = nn.Sigmoid()

        # self.loss_fn = FocalLoss()
        self.loss_fn = nn.CrossEntropyLoss()

        if self.isTrain:

            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)



    def set_input(self, img, label):
        self.inputImg.resize_(img.size()).copy_(img)
        self.label.resize_(label.size()).copy_(label)

    def forward(self):
        self.img = Variable(self.inputImg)
        self.pre = self.net(self.img)
        self.probs = self.sigmoid(self.pre)

    def probability(self,logits):
        return self.sigmoid(logits)

    def backward(self):

        self.probs = self.probability(self.pre)

        self.loss = self.loss_fn(self.probs, self.label)
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
    def save_network(self, network, epoch_label, gpu_ids,best_f1):
        save_filename = 'DenseNet_epoch%s_F1_%f.pth' % (epoch_label,best_f1)
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(network.cpu().state_dict(), save_path)
        # if len(gpu_ids) and torch.cuda.is_available():
        #     network.cuda(device=0)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        self.scheduler.step()
