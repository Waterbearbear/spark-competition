# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random

import cv2
import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image, ImageFile
import numpy as np
from structure import construct_studies
from utils.imgread import get_info, dicom2array,CreatAxialDataset
from utils.util import generate_target

import config

ImageFile.LOAD_TRUNCATED_IMAGES = True


#
# valiPath = r'E:\BME\competition\spark\data\lumbar_train51'
# valijsonPath = r'E:\BME\competition\spark\data\lumbar_train51_annotation.json'
#
# trainPath = r'E:\BME\competition\spark\data\lumbar_train150'
# trainjsonPath = r'E:\BME\competition\spark\data\lumbar_train150_annotation.json'



class sparkset(data.Dataset):
    """
    """

    def __init__(self, data_root_path, data_json_path, is_train=True, transform=None, is_flip=True):
        # specify annotation file for dataset
        if is_train:
            self.data_root_path = data_root_path
            self.data_json_path = data_json_path
        else:
            # 测试集情况待写
            pass
            # self.csv_file = cfg.DATASET.TESTSET
            # self.data_root = cfg.DATASET.TESTROOT

        self.all_data_dict = get_info(self.data_root_path, self.data_json_path)
        self.is_train = is_train
        self.transform = transform
        self.is_flip = is_flip
        # self.data_root = cfg.DATASET.ROOT
        self.input_size = [256, 256]
        self.output_size = [256, 256]
        self.sigma = 1.5
        # self.scale_factor = cfg.DATASET.SCALE_FACTOR

        # rot_factor  （旋转角度)
        # self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = 'Gaussian'
        # self.flip = cfg.DATASET.FLIP
        # self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        # self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.mean = np.array([0.485], dtype=np.float32)
        self.std = np.array([0.229], dtype=np.float32)
        # load annotations
        # self.landmarks_frame = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.all_data_dict)

    def __getitem__(self, idx):
        # idx 为图片索引

        # image_path = os.path.join(self.data_root,
        #                           self.landmarks_frame.iloc[idx, 0])
        # print("img idx : %d"%idx)

        output_size = [256, 256]
        input_size = [256, 256]
        sigma = 1.25

        # result = get_info(trainPath, trainjsonPath)  #获取图片路径及对应的annotation

        # result[:][:] = result
        # result = np.squeeze(result)
        # print(result[0])
        # print(result[0][1][0]['data']['point'][0]['tag'])
        # print(result[0][0])
        # print(len(result))
        # print(type(result))
        # print(type(result))
        # print(result)

        shape_list = []

        # img = cv2.imread(r"E:\BME\HRNet-Facial-Landmark-Detection-master\CEP\images\TrainData\051.bmp")

        # print(type(img))
        # print(img.shape)
        img_dir = self.all_data_dict[idx][0]  # 获取图片的地址
        # print(img_dir)
        img = dicom2array(img_dir)  # 获取具体的图片数据，二维数据
        annotation = self.all_data_dict[idx][1]  # 获取图片的标签
        points = annotation[0]['data']['point']
        pixel_spacing = self.all_data_dict[idx][2]  # 获取pixel-spacing
        pixel_spacing = np.array(pixel_spacing.split('\\'), dtype='float')
        pts = []
        labels = []
        for point in points:
            # print("point", point)
            coord = point['coord']
            tag = point["tag"]
            idc = ide_dict[tag['identification']]
            # center_x, center_y =
            coord_xy = [coord[0], coord[1], idc]
            pts.append(coord_xy)

            if "disc" in tag.keys():
                # print("tag: ",tag)
                label = tag['disc']

            if "vertebra" in tag.keys():
                label = tag["vertebra"]

            # print("label： ",label)
            labels.append(label)
            # identification = tag["tag"]['identification']

            # print(identification)
            # print(tag)
        pts = sorted(pts, key=lambda x: x[2])
        pts = np.array(pts)
        pts = pts[:, :2]
        pts = pts.astype('float').reshape(-1, 2)
        # print("pts.shape: ",pts.shape)
        # print("labels: ",labels)
        # print(type(img))
        # img = img
        # 坐标也要随之变化

        height, weight = img.shape[0], img.shape[1]
        ori_size = min(height, weight)
        if random.random() < 0.5 and self.is_flip:
            img = np.fliplr(img)
            pts = np.array([[weight - i[0], i[1]] for i in pts])
        # print(img.shape)
        # print("before: ", img.shape)
        if weight > height:
            # 必须要用逗号索引
            img = img[:, :height]

        elif height > weight:
            img = img[:weight, :]
        # print(np.shape(img))
        # cv2.imshow('', img)
        # cv2.waitKey(0)

        img = Image.fromarray(img)
        img = img.resize((input_size[0], input_size[1]))
        pts *= [input_size[0] / ori_size, input_size[1] / ori_size]

        # print("img.size: ",img.size)

        # scale *= 1.25
        nparts = pts.shape[0]  # Landmark的数量
        if nparts != 11:
            print(img_dir)
        # print("nparts: ",nparts)

        target = np.zeros((nparts, output_size[0], output_size[1]))
        ##target 为生成的HeatMap   shape:11 x 64 x 64

        # 自己添加的坐标变换，读入的明明是原图的坐标 但是generate heatmap的时候却用了64x64下的坐标
        # 是在transform_pixel的时候处理了
        tpts = pts.copy() * [float(output_size[0]) / float(input_size[0]),
                             float(output_size[0]) / float(input_size[0])]
        ##复制一份Landmark坐标点
        ##transformed points

        for i in range(nparts):
            # 逐个坐标点遍历
            if tpts[i, 1] > 0:
                # 如果y坐标>0 ?
                # 将Landmark进行对应的角度变化和 scale的变化 ?
                # tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                #                                scale, self.output_size, rot=r)

                # tpts[i, 0:2] = transform_pixel(tpts[i, 0:2] + 1,
                #                                self.output_size)

                # 生成heatmap图 target[i] 表示第i个点的Heatmap
                # sigma为超参数
                # 在这部分可能有精度的损失,每次产生的Heatmap不同
                target[i] = generate_target(target[i], tpts[i] - 1, sigma,
                                            label_type=self.label_type)
                # pass
        #
        # print("after: ", img.shape)

        # if img.shape[0] != img.shape[1]:
        #     print("false ",img.shape[0],img.shape[1])
        # else:
        #     print('True', img.shape[0],img.shape[1])
        # if len(img.shape) == 2:
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #
        # shape_list.append(img.shape)
        img = np.asarray(img)
        img = (img / 255.0 - self.mean) / self.std
        img = torch.Tensor(img)
        img = img.unsqueeze(0)
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)  # 转化后所有的的坐标值
        # center = torch.Tensor(center) #当前

        # meta = {'index': idx, 'center': center, 'scale': scale,
        #         'pts': torch.Tensor(pts), 'tpts': tpts, 'box_size': box_size}

        meta = {'index': idx, 'pts': torch.Tensor(pts), 'tpts': tpts, 'ori': ori_size, 'ps': pixel_spacing}

        return img, target, meta


class TestDataset(data.Dataset):
    def __init__(self, test_path):
        self.test_studies = construct_studies(test_path)
        self.test_list = []
        for study in self.test_studies.values():
            self.test_list.append(study.t2_sagittal_middle_frame)
        self.mean = np.array([0.485], dtype=np.float32)
        self.std = np.array([0.229], dtype=np.float32)

    def __len__(self):
        return len(self.test_studies)

    def __getitem__(self, item):
        middle_frame = self.test_list[item]
        img = middle_frame.image
        img = np.asarray(img)
        height, weight = img.shape[0], img.shape[1]
        ori_size = min(height, weight)
        # print(img.shape)
        # print("before: ", img.shape)
        if weight > height:
            # 必须要用逗号索引
            img = img[:, :height]

        elif height > weight:
            img = img[:weight, :]
        else:
            img = img
        input_size = [256, 256]
        img = Image.fromarray(img)
        img = img.resize((input_size[0], input_size[1]))
        img = np.asarray(img)
        img = (img / 255.0 - self.mean) / self.std
        img = torch.Tensor(img)
        img = img.unsqueeze(0)
        meta = {'ori': ori_size, 'study_uid': middle_frame.study_uid, 'series_uid': middle_frame.series_uid,
                'instance_uid': middle_frame.instance_uid, 'path': middle_frame.file_path,
                'instance_number': middle_frame.instance_number}
        return img, meta

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random

import cv2
import torch
import torch.utils.data as data
import torch.nn.functional as F
import pandas as pd
from PIL import Image, ImageFile
import numpy as np

from utils.imgread import get_info, dicom2array,CreatAxialDataset
from utils.util import generate_target

import config
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


#
# valiPath = r'E:\BME\competition\spark\data\lumbar_train51'
# valijsonPath = r'E:\BME\competition\spark\data\lumbar_train51_annotation.json'
#
# trainPath = r'E:\BME\competition\spark\data\lumbar_train150'
# trainjsonPath = r'E:\BME\competition\spark\data\lumbar_train150_annotation.json'



class sparkset(data.Dataset):
    """
    """
    def __init__(self, data_root_path, data_json_path, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.data_root_path = data_root_path
            self.data_json_path = data_json_path
        else:
            # 测试集情况待写
            pass
            # self.csv_file = cfg.DATASET.TESTSET
            # self.data_root = cfg.DATASET.TESTROOT

        self.all_data_dict = get_info(self.data_root_path, self.data_json_path)
        self.is_train = is_train
        self.transform = transform
        # self.data_root = cfg.DATASET.ROOT
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.sigma = 1.5
        # self.scale_factor = cfg.DATASET.SCALE_FACTOR

        # rot_factor  （旋转角度)
        # self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = 'Gaussian'
        # self.flip = cfg.DATASET.FLIP
        # self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        # self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.mean = np.array([0.485], dtype=np.float32)
        self.std = np.array([0.229], dtype=np.float32)
        # load annotations
        # self.landmarks_frame = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.all_data_dict)

    def __getitem__(self, idx):


        # print(type(img))
        # print(img.shape)
        self.img_dir = self.all_data_dict[idx][0]  # 获取图片的地址
        # print(self.img_dir)
        img = dicom2array(self.img_dir)  # 获取具体的图片数据，二维数据
        annotation = self.all_data_dict[idx][1]  # 获取图片的标签
        points = annotation[0]['data']['point']
        pts = []
        labels = []
        for point in points:
            # print("point",point)
            coord = point['coord']
            tag = point["tag"]
            # center_x, center_y =
            coord_xy = [coord[0], coord[1]]
            pts.append(coord_xy)

            if "disc" in tag.keys():
                # print("tag: ",tag)
                label = tag['disc']

            if "vertebra" in tag.keys():
                label = tag["vertebra"]

            # print("label： ",label)
            labels.append(label)
            # identification = tag["tag"]['identification']

            # print(identification)
            # print(tag)
        pts = np.array(pts)
        pts = pts.astype('float').reshape(-1, 2)
        # print("pts.shape: ",pts.shape)
        # print("labels: ",labels)
        # print(type(img))
        # img = img
        # 坐标也要随之变化

        height, weight = img.shape[0], img.shape[1]

        # print(img.shape)
        # print("before: ", img.shape)
        if weight > height:

            extra_weight = weight - height
            crop_left = int(extra_weight / 2) - 1
            crop_right = int(extra_weight / 2) + height - 1

            pts = pts - [0, crop_left]
            # 必须要用逗号索引
            img = img[:, crop_left:crop_right]


        elif height > weight:
            extra_height = height - weight
            crop_up = int(extra_height / 2) - 1
            crop_down = int(extra_height / 2) + weight - 1

            pts = pts - [crop_up, 0]

            img = img[crop_up:crop_down, :]

        img = Image.fromarray(img)
        img = img.resize((self.input_size[0], self.input_size[1]))
        pts *= [self.output_size[0] / self.input_size[0], self.output_size[1] / self.input_size[1]]

        # print("img.size: ",img.size)

        # scale *= 1.25
        nparts = pts.shape[0]  # Landmark的数量
        if nparts != 11:
            print(self.img_dir)
        # print("nparts: ",nparts)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        ##target 为生成的HeatMap   shape:19 x 64 x 64

        # 自己添加的坐标变换，读入的明明是原图的坐标 但是generate heatmap的时候却用了64x64下的坐标
        # 是在transform_pixel的时候处理了
        tpts = pts.copy() * [float(self.output_size[0]) / float(self.input_size[0]),
                             float(self.output_size[0]) / float(self.input_size[0])]
        ##复制一份Landmark坐标点
        ##transformed points

        for i in range(nparts):
            # 逐个坐标点遍历
            if tpts[i, 1] > 0:
                # 如果y坐标>0 ?
                # 将Landmark进行对应的角度变化和 scale的变化 ?
                # tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                #                                scale, self.output_size, rot=r)

                # tpts[i, 0:2] = transform_pixel(tpts[i, 0:2] + 1,
                #                                self.output_size)

                # 生成heatmap图 target[i] 表示第i个点的Heatmap
                # sigma为超参数
                # 在这部分可能有精度的损失,每次产生的Heatmap不同
                target[i] = generate_target(target[i], tpts[i] - 1, config.sigma,
                                            label_type=self.label_type)
                # pass
        #
        # print("after: ", img.shape)

        # if img.shape[0] != img.shape[1]:
        #     print("false ",img.shape[0],img.shape[1])
        # else:
        #     print('True', img.shape[0],img.shape[1])
        # if len(img.shape) == 2:
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #
        # shape_list.append(img.shape)
        # img = np.asarray(img)
        #
        # # img = (img / 255.0 - self.mean) / self.std
        # img = torch.Tensor(img)

        if self.transform:
            img = self.transform(img)

        # img = img.unsqueeze(0)
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)  # 转化后所有的的坐标值
        # center = torch.Tensor(center) #当前

        # meta = {'index': idx, 'center': center, 'scale': scale,
        #         'pts': torch.Tensor(pts), 'tpts': tpts, 'box_size': box_size}

        meta = {'index': idx, 'pts': torch.Tensor(pts), 'tpts': tpts}

        return img, target, meta


class axialdataset(data.Dataset):
    """
    """

    def __init__(self, data_root_path, data_json_path, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.data_root_path = data_root_path
            self.data_json_path = data_json_path
        else:
            # 测试集情况待写
            pass
            # self.csv_file = cfg.DATASET.TESTSET
            # self.data_root = cfg.DATASET.TESTROOT

        self.all_data_dict,self.all_data_csv = CreatAxialDataset(self.data_root_path, self.data_json_path)
        self.is_train = is_train
        self.transform = transform
        self.totensor = transforms.ToTensor()
        # self.data_root = cfg.DATASET.ROOT
        self.input_size = config.input_size
        self.num_classes = config.num_classes

        self.counter = [0,0,0,0,0]

        self.map = {'v1': 0, 'v2': 1, 'v3': 2, 'v4': 3, 'v5': 4}

        self.mean = np.array([0.485], dtype=np.float32)
        self.std = np.array([0.229], dtype=np.float32)

    def __len__(self):
        return len(self.all_data_dict)

    def __getitem__(self, idx):
        # idx 为图片索引

        shape_list = []

        # print(type(img))
        # print(img.shape)
        self.img_dir = self.all_data_dict[idx]['dcmPath']  # 获取图片的地址

        # print(self.img_dir)
        try:
            img = dicom2array(self.img_dir)  # 获取具体的图片数据，二维数据
        except:
            print("read dirty data: ",self.img_dir)

        label = self.all_data_dict[idx]['label']  # 获取图片的标签

        # 有些label为两个值,选择前一个值
        if ',' in label:
            label = label.split(',')[0]

        label = self.map[label]

        height, weight = img.shape[0], img.shape[1]

        # print(img.shape)
        # print("before: ", img.shape)

        # 图像裁剪
        if weight > height:

            extra_weight = weight - height
            crop_left = int(extra_weight / 2) - 1
            crop_right = int(extra_weight / 2) + height - 1

            img = img[:, crop_left:crop_right]


        elif height > weight:
            extra_height = height - weight
            crop_up = int(extra_height / 2) - 1
            crop_down = int(extra_height / 2) + weight - 1
            img = img[crop_up:crop_down, :]

        img = Image.fromarray(img)

        img = img.resize((self.input_size[0], self.input_size[1]))
        # print(img.size)


        # shape_list.append(img.shape)

        if self.transform is not None and label != 0:
            img = self.transform(img)

        img = self.totensor(img)

        # print("self.img_dir: ",self.img_dir)
        # print("label: ",label)

        # img = img.unsqueeze(0)
        # print("type label: ",type(label))

        return img,label




if __name__ == '__main__':
    pass
if __name__ == '__main__':
    pass
