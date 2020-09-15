import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from ..backbone.net import NestedUNet, UNet_3Plus


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
        self.inputImg = self.Tensor(opt.batchsize, 3, 256, 256)
        self.target = self.Tensor(opt.batchsize, 3, 256, 256)
        self.preCoord = []
        self.net = UNet_3Plus(in_channels=3, n_classes=11).cuda()
        self.loss_fn = nn.MSELoss()
        if self.isTrain:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
            if opt.continue_train:
                self.load_network(self.net, 150)
        else:
            self.load_network(self.net, opt.weights)

    def set_input(self, img, target, meta):
        self.inputImg.resize_(img.size()).copy_(img)
        self.target.resize_(target.size()).copy_(target)
        self.meta = meta

    def set_pre_input(self, img, meta):
        self.inputImg.resize_(img.size()).copy_(img)
        self.meta = meta

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

    def getHeatmap(self):
        self.preHeatmap = self.pre.cpu().detach().numpy()
        self.tarHeatmap = self.target.cpu().detach().numpy()
        return self.tarHeatmap, self.preHeatmap

    def get_preds(self, scores):
        """
        get predictions from score maps in torch Tensor
        return type: torch.LongTensor
        """
        assert scores.dim() == 4, 'Score maps should be 4-dim'
        maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

        maxval = maxval.view(scores.size(0), scores.size(1), 1)
        idx = idx.view(scores.size(0), scores.size(1), 1) + 1

        preds = idx.repeat(1, 1, 2).float()

        preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
        preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

        pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
        preds *= pred_mask
        return preds

    def getPreCoord(self):
        self.preHeatmap = self.pre.cpu().detach().numpy()
        heatlist = self.preHeatmap[0, :, :, :]
        ori_size = self.meta['ori']
        prePts = []
        for i in range(11):
            index = np.where(heatlist[i] == np.max(heatlist[i]))
            prePts.append((index[1].tolist()[0], index[0].tolist()[0]))
        prePts = np.array(prePts)
        preCoord = prePts.copy() * (float(ori_size) / float(256), float(ori_size) / float(256))
        return prePts, preCoord

    # used in test time, no backprop
    def test(self):
        self.img = Variable(self.inputImg, requires_grad=False)
        self.pre = self.net(self.img)

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
    def load_network(self, network, epoch_label):
        save_filename = '%s.pth' % epoch_label
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        self.scheduler.step()


class SparkModel(nn.Module):
    def name(self):
        return 'Model'

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.use_bf16 = opt.use_bf16
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir)
        self.preCoord = []
        self.net = UNet_3Plus(in_channels=3, n_classes=11)
        self.loss_fn = nn.MSELoss()
        if self.isTrain:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
            if opt.continue_train:
                self.load_network(self.net, 150)
        else:
            self.load_network(self.net, opt.weights)

    def set_input(self, img, target, meta):
        self.inputImg.resize_(img.size()).copy_(img)
        self.target.resize_(target.size()).copy_(target)
        self.meta = meta

    def set_pre_input(self, img, meta):
        self.inputImg.resize_(img.size()).copy_(img)
        self.meta = meta

    def forward(self, args):
        if self.training:
            self._train(*args)

    def _train(self, img, target):
        if self.use_bf16:
            img = img.to_mkldnn(torch.bfloat16)
        pre = self.net(img)
        loss = self.loss_fn(target, pre)
        return loss

    def backward(self):
        self.loss = self.loss_fn(self.target, self.pre)
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def getHeatmap(self):
        self.preHeatmap = self.pre.cpu().detach().numpy()
        self.tarHeatmap = self.target.cpu().detach().numpy()
        return self.tarHeatmap, self.preHeatmap

    def get_preds(self, scores):
        """
        get predictions from score maps in torch Tensor
        return type: torch.LongTensor
        """
        assert scores.dim() == 4, 'Score maps should be 4-dim'
        maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

        maxval = maxval.view(scores.size(0), scores.size(1), 1)
        idx = idx.view(scores.size(0), scores.size(1), 1) + 1

        preds = idx.repeat(1, 1, 2).float()

        preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
        preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

        pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
        preds *= pred_mask
        return preds

    def getPreCoord(self):
        self.preHeatmap = self.pre.cpu().detach().numpy()
        heatlist = self.preHeatmap[0, :, :, :]
        ori_size = self.meta['ori']
        prePts = []
        for i in range(11):
            index = np.where(heatlist[i] == np.max(heatlist[i]))
            prePts.append((index[1].tolist()[0], index[0].tolist()[0]))
        prePts = np.array(prePts)
        preCoord = prePts.copy() * (float(ori_size) / float(256), float(ori_size) / float(256))
        return prePts, preCoord

    # used in test time, no backprop
    def test(self):
        self.img = Variable(self.inputImg, requires_grad=False)
        self.pre = self.net(self.img)

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
    def load_network(self, network, epoch_label):
        save_filename = '%s.pth' % epoch_label
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        self.scheduler.step()


class NullLoss:
    def __call__(self, x, y):
        return x[0].mean()
