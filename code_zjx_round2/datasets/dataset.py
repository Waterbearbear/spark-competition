# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random
import copy
import cv2
import torch
import torch.utils.data as data
import torch.nn.functional as F
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFile
import numpy as np
import joblib
from utils.imgread import get_info, dicom2array,CreatAxialDataset
from utils.util import generate_target
from structure.study import construct_studies
import config
from torchvision import transforms
from utils.imgread import CaculateDisc3DCoordinate,PointToSurfaceDistance

from utils.imgread import CreatCropSagDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


#
# valiPath = r'E:\BME\competition\spark\data\lumbar_train51'
# valijsonPath = r'E:\BME\competition\spark\data\lumbar_train51_annotation.json'
#
# trainPath = r'E:\BME\competition\spark\data\lumbar_train150'
# trainjsonPath = r'E:\BME\competition\spark\data\lumbar_train150_annotation.json'



class PrepareCropData:
    def __init__(self, data_path, annotation_path,phase):
        pkl_path_temp = ['studies_%s.pkl'%phase, 'annotation_%s.pkl'%phase,
                    'vertebra_%s.pkl'%phase, 'disc_%s.pkl'%phase]

        pkl_path = [os.path.join(config.external_data_path,i) for i in pkl_path_temp]


        if os.path.exists(pkl_path[2]) and os.path.exists(pkl_path[3]):
            print("exist!")

            self.vertebra_data = joblib.load(pkl_path[2])
            self.disc_data = joblib.load(pkl_path[3])
        else:
            self.middle_list = []
            if not (os.path.exists(pkl_path[0]) and os.path.exists(pkl_path[1])):
                self.studies, self.annotation, _ = construct_studies(data_path, annotation_path)
                joblib.dump(self.studies, pkl_path[0])
                joblib.dump(self.annotation, pkl_path[1])
            else:
                self.studies = joblib.load(pkl_path[0])
                self.annotation = joblib.load(pkl_path[1])
            for study in self.studies.values():
                self.middle_list.append(study.t2_sagittal_middle_frame)
            self.vertebra_data, self.disc_data = self.get_all_item()
            joblib.dump(self.vertebra_data, pkl_path[2])
            joblib.dump(self.disc_data, pkl_path[3])

    def get_all_item(self):
        vertebra_data = {}
        disc_data = {}
        vertebra_map = ['L1', 'L2', 'L3', 'L4', 'L5']
        disc_map = ['T12-L1', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']
        for iter, middle_frame in enumerate(self.middle_list):
            vertebra_data[middle_frame.study_uid] = {}
            disc_data[middle_frame.study_uid] = {}
            annotation = self.annotation[middle_frame.study_uid, middle_frame.series_uid, middle_frame.instance_uid]
            coord_vertebra = annotation[0][:, :2]
            print(coord_vertebra.shape)
            coord_disc = annotation[1][:, :2]
            label_vertebra = annotation[0][:, 2]
            label_disc = annotation[1][:, 2]
            img = middle_frame.image
            img = np.asarray(img)
            height, width = img.shape[0], img.shape[1]
            img = cv2.resize(img, (512, 512))
            # cv2.imshow('', img)
            # cv2.waitKey(0)
            coord_vertebra = coord_vertebra.numpy() * [float(512) / float(width),
                                                       float(512) / float(height)]
            coord_disc = coord_disc.numpy() * [float(512) / float(width),
                                               float(512) / float(height)]
            #  crop_size = (48, 96)
            for i, coord in enumerate(coord_vertebra):
                print(i)
                crop_img = img[int(coord[1]) - 24:int(coord[1]) + 24, int(coord[0]) - 48: int(coord[0]) + 48]
                print(crop_img.shape)
                if crop_img.shape != (48, 96):
                    cv2.imshow('', img)
                    cv2.waitKey(0)
                crop_dict = {'img': crop_img, 'label': label_vertebra[i]}
                vertebra_data[middle_frame.study_uid][vertebra_map[i]] = crop_dict
            for i, coord in enumerate(coord_disc):
                crop_img = img[int(coord[1]) - 24:int(coord[1]) + 24, int(coord[0]) - 48: int(coord[0]) + 48]
                if crop_img.shape != (48, 96):
                    cv2.imshow('', img)
                    cv2.waitKey(0)
                crop_dict = {'img': crop_img, 'label': label_disc[i]}
                disc_data[middle_frame.study_uid][disc_map[i]] = crop_dict
        # vertebra_data = {study_uid: {'T12': {'img': np.array, 'label': tensor}, ...}, ...}
        # disc_data = {study_uid: {'T12-L1': {'img': np.array, 'label': tensor}, ...}, ...}
        return vertebra_data, disc_data


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
        self.data_root_path = data_root_path
        self.data_json_path = data_json_path
        self.is_train = is_train

        # self.csv_file = cfg.DATASET.TESTSET
        # self.data_root = cfg.DATASET.TESTROOT

        self.all_data_dict,self.all_data_csv = CreatAxialDataset(self.data_root_path,self.data_json_path,is_train = self.is_train)
        # print(self.all_data_dict)
        # print(self.all_data_csv)
        # print(is_train)

        self.transform = transform
        self.totensor = transforms.ToTensor()
        # self.data_root = cfg.DATASET.ROOT
        self.input_size = config.input_size
        self.num_classes = config.num_classes

        self.counter = [0,0,0,0,0]

        self.map = {'v1': 0, 'v2': 1, 'v3': 2, 'v4': 3, 'v5': 4,'v1_v':5,'v2_v':6}

        self.mean = np.array([0.485], dtype=np.float32)
        self.std = np.array([0.229], dtype=np.float32)

    def __len__(self):
        return len(self.all_data_dict)

    def __getitem__(self, idx):
        # idx 为图片索引

        shape_list = []

        # print(type(img))
        # print(img.shape)

        # print(self.all_data_dict)
        self.img_dir = self.all_data_dict[idx]['dcmPath']  # 获取图片的地址
        print(self.img_dir)

        # print(self.img_dir)
        try:
            img = dicom2array(self.img_dir)  # 获取具体的图片数据，二维数据
        except:
            print("read dirty data: ",self.img_dir)

        label = self.all_data_dict[idx]['label']  # 获取图片的标签

        # 矢状面的路径
        disc_path = self.all_data_dict[idx]['disc_dcmPath']
        identification = self.all_data_dict[idx]['identification']
        studyID = self.all_data_dict[idx]['studyUid']

        # 有些label为两个值,选择前一个值
        if self.is_train:
            if ',' in label :
                label = label.split(',')[0]
            if identification == 'vertebra':
                label = label + '_v'
            label = self.map[label]

        height, weight = img.shape[0], img.shape[1]

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


        if self.transform is not None and label != 0:
            img = self.transform(img)

        img = self.totensor(img)

        # print("self.img_dir: ",self.img_dir)
        # print("label: ",label)

        # img = img.unsqueeze(0)
        # print("type label: ",type(label))

        if self.is_train:
            return img,label
        else:
            return img,label,disc_path,identification,studyID


class Axial_Testset(data.Dataset):


    def __init__(self, test_dict_path, transform=None):

        self.test_dict_path = test_dict_path
        self.transform = transform
        self.totensor = transforms.ToTensor()

        self.input_size = config.input_size



        self.test_dict = np.load(self.test_dict_path,allow_pickle = True)

    def __len__(self):

        return  len(self.test_dict)

    def __getitem__(self, idx):

        self.img_dir = self.test_dict[idx]['dcmPath']  # 获取图片的地址

        # print(self.img_dir)
        try:
            img = dicom2array(self.img_dir)  # 获取具体的图片数据，二维数据
        except:
            print("read dirty data: ", self.img_dir)


        axial_path = self.test_dict[idx]['axial_path']
        identification = self.test_dict[idx]['identification']
        studyID = self.test_dict[idx]['studyUid']

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

        img = self.totensor(img)

        # print("self.img_dir: ",self.img_dir)
        # print("label: ",label)

        # img = img.unsqueeze(0)
        # print("type label: ",type(label))


        return img, axial_path,identification, studyID

class SagAxialDataset(data.Dataset):
    """
    """
    def __init__(self, data_root_path,data_json_path, part,is_train=True, transform=None):
        # data_root_path : 数据所在的路径
        # data_json_path : 数据标记所在路径
        # pkl_path       : sag切下来的锥体和椎间盘的pkl文件路径
        # part           : 当前数据集用于提取"vertebra" 还是 "disc"
        # is_train       : "all",True,False. 用于表示训练还是测试状态
        # transform      : 数据增强

        self.disc_name = []
        self.vertebra_name = []
        # specify annotation file for dataset
        self.data_root_path = data_root_path
        self.data_json_path = data_json_path
        # self.test_dict_path = test_dict_path
        # self.test_csv_path  = test_csv_path

        self.is_train = is_train
        self.part     = part
        self.vertebra_list = ['L1', 'L2', 'L3', 'L4', 'L5']
        self.disc_list = ['T12-L1', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']


        # self.csv_file = cfg.DATASET.TESTSET
        # self.data_root = cfg.DATASET.TESTROOT

        self.all_axialdata_dict,self.all_axialdata_csv = CreatAxialDataset(self.data_root_path,self.data_json_path,is_train = self.is_train)
        # self.test_dict = np.load(self.test_dict_path,allow_pickle = True)
        # self.test_csv  = pd.read_csv(self.test_csv_path)

        if is_train:
            sag_train = PrepareCropData(config.trainPath, config.trainjsonPath, "train")
            sag_val = PrepareCropData(config.valPath, config.valjsonPath, "val")

            self.sag_total = copy.deepcopy(sag_train)

            # print(type(pre_train.disc_data))

            ##150和51的数据合成为一个
            for key, study in sag_val.disc_data.items():
                self.sag_total.disc_data[key] = study

            for key, study in sag_val.vertebra_data.items():
                self.sag_total.vertebra_data[key] = study



        else:
            self.sag_total = PrepareCropData(config.testPath,config.testjsonPath,"test")


        if self.part == 'vertebra':
            self.part_data_csv = self.all_axialdata_csv.loc[self.all_axialdata_csv['identification'].isin(self.vertebra_list), :]
            #
            # self.part_data_csv = self.test_csv.loc[self.test_csv['vertebra'] == 1,:]
        elif self.part == 'disc':
            self.part_data_csv = self.all_axialdata_csv.loc[self.all_axialdata_csv['identification'].isin(self.disc_list), :]
            # self.part_data_csv = self.test_csv.loc[self.test_csv['disc'] == 1,:]
        else:
            print("part value should be  'disc' or 'vertebra' ")
            raise ValueError

        self.part_data_csv.reset_index(drop = True,inplace = True)
        self.part_data_dict = self.part_data_csv.to_dict(orient='records')
        self.map = {'v1': 0, 'v2': 1, 'v3': 2, 'v4': 3, 'v5': 4}
        self.transform = transform
        self.totensor = transforms.ToTensor()
        # self.data_root = cfg.DATASET.ROOT
        self.input_size = config.input_size
        self.num_classes = config.num_classes

        self.counter = [0,0,0,0,0]

    def __len__(self):
        return len(self.part_data_csv)

    def __getitem__(self, idx):
        # idx 为图片索引

        # print(type(img))
        # print(img.shape)

        # print(self.all_data_dict)
        disc_path = self.part_data_dict[idx]['disc_dcmPath']
        # disc_path = self.part_data_dict[idx]['dcmPath']
        identification = self.part_data_dict[idx]['identification']
        studyID = self.part_data_dict[idx]['studyUid']
        label = self.part_data_dict[idx]['label']  # 获取图片的标签

        self.axial_img_dir = self.part_data_dict[idx]['dcmPath']  # 获取图片的地址

        # print(studyID)
        # print(identification)

        try:
            axial_img = dicom2array(self.axial_img_dir)  # 获取具体的图片数据，二维数据
        except:
            print("read dirty data: ",self.axial_img_dir)


        if self.part == 'vertebra':
            sag_img = self.sag_total.vertebra_data[studyID][identification]['img']
        elif self.part == 'disc':
            sag_img = self.sag_total.disc_data[studyID][identification]['img']
        else:
            raise ValueError

        # print("idx: %d"%idx)
        # print(axial_img)
        # print("shape before:",axial_img.shape)
        # print(self.axial_img_dir)
        # print(studyID)
        # 矢状面的路径

        # 有些label为两个值,选择前一个值
        if self.is_train:
            if ',' in label :
                label = label.split(',')[0]
            label = self.map[label]


        axial_height, axial_weight = axial_img.shape[0], axial_img.shape[1]
        sag_height , sag_weight    = sag_img.shape[0] , sag_img.shape[1]
        # print(img.shape)
        # print("before: ", img.shape)

        # 图像裁剪
        if axial_weight > axial_height:

            extra_weight = axial_weight - axial_height
            crop_left = int(extra_weight / 2) - 1
            if crop_left < 0:
                crop_left = 0

            crop_right = int(extra_weight / 2) + axial_height - 1


            axial_img = axial_img[:, crop_left:crop_right]


        elif axial_height > axial_weight:
            extra_height = axial_height - axial_weight
            crop_up = int(extra_height / 2) - 1
            if crop_up < 0:
                crop_up = 0
            crop_down = int(extra_height / 2) + axial_weight - 1
            axial_img = axial_img[crop_up:crop_down, :]



        # print("shape after:",axial_img.shape)
        #
        # print("sag img:")
        # print(sag_img)
        # print(sag_img.shape)

        axial_img = Image.fromarray(axial_img)
        sag_img   = Image.fromarray(sag_img)

        if sag_weight > sag_height:

            self.padding = transforms.Pad(padding = (0,int((sag_weight - sag_height)/2)))

        else:
            self.padding = transforms.Pad(padding= (int((sag_height - sag_weight)/2) , 0) )

        sag_img = self.padding(sag_img)

        # print("shape PIL")
        # print(axial_img)

        # axial_img = axial_img.resize((self.input_size[0], self.input_size[1]))
        # print(img.size)

        # shape_list.append(img.shape)

        if self.transform is not None:
            axial_img = self.transform(axial_img)
            sag_img = self.transform(sag_img)

        # img = self.totensor(img)

        # print("self.img_dir: ",self.img_dir)
        # print("label: ",label)

        # img = img.unsqueeze(0)
        # print("type label: ",type(label))


        return axial_img,sag_img,label,disc_path,identification,studyID


class CropSagDataset(data.Dataset):
    """
         数据路径在config中指定
         pkl_path       : sag切下来的锥体和椎间盘的pkl文件路径
         part           : 当前数据集用于提取"vertebra" 还是 "disc"
         is_train       : "all",True,False. 用于表示训练还是测试状态
         transform      : 数据增强

    """
    def __init__(self, part,is_train=True, transform=None):



        self.disc_name = []
        self.vertebra_name = []
        self.is_train = is_train
        self.part     = part
        self.vertebra_list = ['L1', 'L2', 'L3', 'L4', 'L5']
        self.disc_list = ['T12-L1', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']

        if is_train:
            sag_train = PrepareCropData(config.trainPath, config.trainjsonPath, "train")
            sag_val = PrepareCropData(config.valPath, config.valjsonPath, "val")

            self.sag_total = copy.deepcopy(sag_train)

            # print(type(pre_train.disc_data))

            ##150和51的数据合成为一个
            for key, study in sag_val.disc_data.items():
                self.sag_total.disc_data[key] = study

            for key, study in sag_val.vertebra_data.items():
                self.sag_total.vertebra_data[key] = study

        else:
            self.sag_total = PrepareCropData(config.testPath,config.testjsonPath,"test")

        self.all_crop_sag = CreatCropSagDataset(self.sag_total,part)
        # sag_vertebra = CreatCropSagDataset(sag_total, 'vertebra')
        # if self.part == 'vertebra':
        #     self.part_data_csv = self.all_axialdata_csv.loc[self.all_axialdata_csv['identification'].isin(self.vertebra_list), :]
        #     #
        #     # self.part_data_csv = self.test_csv.loc[self.test_csv['vertebra'] == 1,:]
        # elif self.part == 'disc':
        #     self.part_data_csv = self.all_axialdata_csv.loc[self.all_axialdata_csv['identification'].isin(self.disc_list), :]
            # self.part_data_csv = self.test_csv.loc[self.test_csv['disc'] == 1,:]
        # else:
        #     print("part value should be  'disc' or 'vertebra' ")
        #     raise ValueError

        # self.part_data_csv.reset_index(drop = True,inplace = True)
        # self.part_data_dict = self.part_data_csv.to_dict(orient='records')

        self.map = {'v1': 0, 'v2': 1, 'v3': 2, 'v4': 3, 'v5': 4}
        self.transform = transform
        self.totensor = transforms.ToTensor()
        # self.data_root = cfg.DATASET.ROOT
        self.input_size = config.input_size
        self.num_classes = config.num_classes

        self.counter = [0,0,0,0,0]

    def __len__(self):
        return len(self.all_crop_sag)

    def __getitem__(self, idx):
        # idx 为图片索引

        # print(type(img))
        # print(img.shape)

        # print(self.all_data_dict)
        # disc_path = self.part_data_dict[idx]['disc_dcmPath']
        # disc_path = self.part_data_dict[idx]['dcmPath']

        crop_sag_img = self.all_crop_sag[idx]['img']
        identification = self.all_crop_sag[idx]['identification']
        studyID = self.all_crop_sag[idx]['studyUid']

        #dtype:tensor
        label = self.all_crop_sag[idx]['label']

        # print("idx: %d"%idx)
        # print(axial_img)
        # print("shape before:",axial_img.shape)
        # print(self.axial_img_dir)
        # print(studyID)
        # 矢状面的路径
        # 有些label为两个值,选择前一个值
        # if self.is_train:
        #     if ',' in label :
        #         label = label.split(',')[0]
        #     label = self.map[label]
        sag_height , sag_weight    = crop_sag_img.shape[0] , crop_sag_img.shape[1]
        # print(img.shape)
        # print("before: ", img.shape)

        # 图像裁剪
        crop_sag_img   = Image.fromarray(crop_sag_img)

        if sag_weight > sag_height:

            self.padding = transforms.Pad(padding = (0,int((sag_weight - sag_height)/2)))

        else:
            self.padding = transforms.Pad(padding= (int((sag_height - sag_weight)/2) , 0) )

        crop_sag_img = self.padding(crop_sag_img)

        # print("shape PIL")
        # print(axial_img)

        # axial_img = axial_img.resize((self.input_size[0], self.input_size[1]))
        # print(img.size)

        # shape_list.append(img.shape)

        if self.transform is not None:
            # axial_img = self.transform(axial_img)
            try:
                crop_sag_img = self.transform(crop_sag_img)
            except:
                print("studyUid: ", studyID)
                print("identification: ", identification)

                print("crop sag img: ")
                print(crop_sag_img)

        # img = self.totensor(img)

        # print("self.img_dir: ",self.img_dir)
        # print("label: ",label)

        # img = img.unsqueeze(0)
        # print("type label: ",type(label))

        # return axial_img, sag_img, label, disc_path, identification, studyID
        axial_img = np.zeros(1)
        disc_path = ''

        return axial_img,crop_sag_img,label,disc_path,identification,studyID


class SagAxial_Test_Dataset(data.Dataset):
    """
    """
    def __init__(self, test_dict_path,test_csv_path, part,is_train=True, transform=None):
        # data_root_path : 数据所在的路径
        # data_json_path : 数据标记所在路径
        # pkl_path       : sag切下来的锥体和椎间盘的pkl文件路径
        # part           : 当前数据集用于提取"vertebra" 还是 "disc"
        # is_train       : "all",True,False. 用于表示训练还是测试状态
        # transform      : 数据增强

        self.disc_name = []
        self.vertebra_name = []
        # specify annotation file for dataset
        # self.data_root_path = data_root_path
        # self.data_json_path = data_json_path
        self.test_dict_path = test_dict_path
        self.test_csv_path  = test_csv_path

        self.is_train = is_train
        self.part     = part
        self.vertebra_list = ['L1', 'L2', 'L3', 'L4', 'L5']
        self.disc_list = ['T12-L1', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']


        # self.csv_file = cfg.DATASET.TESTSET
        # self.data_root = cfg.DATASET.TESTROOT

        # self.all_axialdata_dict,self.all_axialdata_csv = CreatAxialDataset(self.data_root_path,self.data_json_path,is_train = self.is_train)
        self.test_dict = np.load(self.test_dict_path,allow_pickle = True)
        self.test_csv  = pd.read_csv(self.test_csv_path)

        if is_train:
            sag_train = PrepareCropData(config.trainPath, config.trainjsonPath, "train")
            sag_val = PrepareCropData(config.valPath, config.valjsonPath, "val")

            self.sag_total = copy.deepcopy(sag_train)

            # print(type(pre_train.disc_data))

            for key, study in sag_val.disc_data.items():
                self.sag_total.disc_data[key] = study

            for key, study in sag_val.vertebra_data.items():
                self.sag_total.vertebra_data[key] = study

        else:
            self.sag_total = PrepareCropData(config.testPath,config.testjsonPath,"test")


        if self.part == 'vertebra':

            # self.part_data_csv = self.all_axialdata_csv.loc[self.all_axialdata_csv['identification'].isin(self.vertebra_list), :]

            self.part_data_csv = self.test_csv.loc[self.test_csv['vertebra'] == 1,:]
        elif self.part == 'disc':
            # self.part_data_csv = self.all_axialdata_csv.loc[self.all_axialdata_csv['identification'].isin(self.disc_list), :]
            self.part_data_csv = self.test_csv.loc[self.test_csv['disc'] == 1,:]
        else:
            print("part value should be  'disc' or 'vertebra' ")
            raise ValueError

        self.part_data_csv.reset_index(drop = True,inplace = True)
        self.part_data_dict = self.part_data_csv.to_dict(orient='records')
        self.map = {'v1': 0, 'v2': 1, 'v3': 2, 'v4': 3, 'v5': 4}
        self.transform = transform
        self.totensor = transforms.ToTensor()
        # self.data_root = cfg.DATASET.ROOT
        self.input_size = config.input_size
        self.num_classes = config.num_classes

        self.counter = [0,0,0,0,0]

    def __len__(self):
        return len(self.part_data_csv)

    def __getitem__(self, idx):
        # idx 为图片索引

        # print(type(img))
        # print(img.shape)

        # print(self.all_data_dict)
        # disc_path = self.part_data_dict[idx]['disc_dcmPath']
        disc_path = self.part_data_dict[idx]['dcmPath']
        identification = self.part_data_dict[idx]['identification']
        studyID = self.part_data_dict[idx]['studyUid']
        label = self.part_data_dict[idx]['label']  # 获取图片的标签
        self.axial_img_dir = self.part_data_dict[idx]['axial_path']  # 获取图片的地址

        # print(studyID)
        # print(identification)

        try:
            axial_img = dicom2array(self.axial_img_dir)  # 获取具体的图片数据，二维数据
        except:
            print("read dirty data: ",self.axial_img_dir)


        if self.part == 'vertebra':
            sag_img = self.sag_total.vertebra_data[studyID][identification]['img']
        elif self.part == 'disc':
            sag_img = self.sag_total.disc_data[studyID][identification]['img']
        else:
            raise ValueError

        # print("idx: %d"%idx)
        # print(axial_img)
        # print("shape before:",axial_img.shape)
        # print(self.axial_img_dir)
        # print(studyID)
        # 矢状面的路径

        # 有些label为两个值,选择前一个值
        if self.is_train:
            if ',' in label :
                label = label.split(',')[0]
            label = self.map[label]


        axial_height, axial_weight = axial_img.shape[0], axial_img.shape[1]
        sag_height , sag_weight    = sag_img.shape[0] , sag_img.shape[1]
        # print(img.shape)
        # print("before: ", img.shape)



        # 图像裁剪
        if axial_weight > axial_height:

            extra_weight = axial_weight - axial_height
            crop_left = int(extra_weight / 2) - 1
            if crop_left < 0:
                crop_left = 0

            crop_right = int(extra_weight / 2) + axial_height - 1


            axial_img = axial_img[:, crop_left:crop_right]


        elif axial_height > axial_weight:
            extra_height = axial_height - axial_weight
            crop_up = int(extra_height / 2) - 1
            if crop_up < 0:
                crop_up = 0
            crop_down = int(extra_height / 2) + axial_weight - 1
            axial_img = axial_img[crop_up:crop_down, :]



        # print("shape after:",axial_img.shape)
        #
        # print("sag img:")
        # print(sag_img)
        # print(sag_img.shape)

        axial_img = Image.fromarray(axial_img)
        sag_img   = Image.fromarray(sag_img)

        if sag_weight > sag_height:

            self.padding = transforms.Pad(padding = (0,int((sag_weight - sag_height)/2)))

        else:
            self.padding = transforms.Pad(padding= (int((sag_height - sag_weight)/2) , 0) )

        sag_img = self.padding(sag_img)

        # print("shape PIL")
        # print(axial_img)

        # axial_img = axial_img.resize((self.input_size[0], self.input_size[1]))
        # print(img.size)

        # shape_list.append(img.shape)

        if self.transform is not None:
            axial_img = self.transform(axial_img)
            sag_img = self.transform(sag_img)

        # img = self.totensor(img)

        # print("self.img_dir: ",self.img_dir)
        # print("label: ",label)

        # img = img.unsqueeze(0)
        # print("type label: ",type(label))


        return axial_img,sag_img,label,disc_path,identification,studyID


if __name__ == '__main__':
    #################  矢状面图像Crop ###################################


    print(config.trainPath)
    print(config.trainjsonPath)

    print(config.valPath)
    print(config.valjsonPath)

    # pre_train = PrepareCropData(config.trainPath, config.trainjsonPath,"train")



    pre_val   = PrepareCropData(config.valPath, config.valjsonPath,"val")

    print(pre_val)


    # print()

    # pre_total = copy.deepcopy(pre_train)


    # print(type(pre_train.disc_data))

    # for key,study in pre_val.disc_data.items():
    #     pre_total.disc_data[key] = study
    #
    # for key,study in pre_val.vertebra_data.items():
    #     pre_total.vertebra_data[key] = study


    # img = pre_total.vertebra_data['1.2.1114.250756.6.682.420717188774813080248514492603145951793']['L2']['img']

    # plt.figure()
    # plt.imshow(img,cmap = 'gray')
    #
    # sns.distplot(img)
    # plt.show()
    #
    # print(type(img))
    #
    #
    #################  矢状面图像Crop ###################################


  ###################### 提取csv文件测试  ###############


    # all_axialdata_dict, all_axialdata_csv = CreatAxialDataset(config.trainPath, config.trainjsonPath,
    #                                                                         is_train='all')
    #
    # part = 'vertebra'
    #
    # vertebra_list = ['L1','L2','L3','L4','L5']
    # disc_list     = ['T12-L1','L1-L2','L2-L3','L3-L4','L4-L5','L5-S1']
    #
    # ver_part_data_csv = all_axialdata_csv.loc[all_axialdata_csv['identification'].isin(disc_list), :]
    # disc_part_data_csv = all_axialdata_csv.loc[all_axialdata_csv['identification'].isin(vertebra_list), :]
    #
    # ver_part_data_dict = ver_part_data_csv.to_dict(orient= 'records')
    #
    # print(ver_part_data_dict[0]['identification'])
    # print(part_data_csv)
