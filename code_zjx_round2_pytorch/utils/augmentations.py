import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as transforms
from PIL import Image
import random



def horisontal_flip(images, targets):

    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    #Mixup技术,将数据集内图像进行加权融合
    # 权值由alpha控制

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.1:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]  # 自己和打乱的自己进行叠加
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def addnoise(img,sigma = 0.001):

    img = np.array(img)
    # print(img.shape)
    height = img.shape[0]
    width = img.shape[1]
    depth = img.shape[2]
    noise_image = img + np.random.randn(height, width,depth) * sigma

    # print(type(noise_image))
    # print(noise_image.shape)

    noise_image = Image.fromarray(np.uint8(noise_image))

    return  noise_image



class RandomHorizontalLandmarkFlip(object):

    def __init__(self,p = 0.5):
        self.p = p

    def __call__(self, images,targets):
        if random.random() < self.p:
            images = torch.flip(images, [-1])
            targets[:, 2] = 1 - targets[:, 2]

        return images, targets

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomAddGaussianNoise(object):

    def __init__(self,sigma = 0.001,p = 0.5):
        self.p = p
        self.sigma = sigma

    def __call__(self, img):

        if random.random() < self.p:

            img = np.array(img,dtype=float)/255
            noise = np.random.normal(0,1**0.5,img.shape) * self.sigma

            out = img + noise
            if out.min() < 0:
                low_clip = -1
            else:
                low_clip = 0

            out = np.clip(out,low_clip,1.0)
            output = np.uint8(out*255)
            # noise_image = img + np.random.randn(height, width, depth) * self.sigm
            # print(noise_image.shape)
            # print(type(noise_image))
            # print(noise_image.shape)
            noise_image = Image.fromarray(output)

            return noise_image
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)




