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
import json
import torch.utils.data as data
import torch.nn.functional as F
import pandas as pd
from albumentations import CLAHE, Compose, Normalize, HueSaturationValue
from albumentations.pytorch.transforms import ToTensor
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFile
import numpy as np
import joblib
from ..utils.imgread import get_info, dicom2array, CreatAxialDataset
from ..utils.util import generate_target, coordRotate
from ..structure.study import construct_studies
import jizhu.config as config
from torchvision import transforms
from ..utils.imgread import CaculateDisc3DCoordinate, PointToSurfaceDistance

ImageFile.LOAD_TRUNCATED_IMAGES = True


#
# valiPath = r'E:\BME\competition\spark\data\lumbar_train51'
# valijsonPath = r'E:\BME\competition\spark\data\lumbar_train51_annotation.json'
#
# trainPath = r'E:\BME\competition\spark\data\lumbar_train150'
# trainjsonPath = r'E:\BME\competition\spark\data\lumbar_train150_annotation.json'


class PrepareCropData:
    def __init__(self, data_path, annotation_path, phase):
        pkl_path_temp = ['studies_%s.pkl' % phase, 'annotation_%s.pkl' % phase,
                         'vertebra_%s.pkl' % phase, 'disc_%s.pkl' % phase]

        pkl_path = [os.path.join(config.external_data_path, i) for i in pkl_path_temp]

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
                # print(i)
                crop_img = img[int(coord[1]) - 24:int(coord[1]) + 24, int(coord[0]) - 48: int(coord[0]) + 48]
                # print(crop_img.shape)
                # if crop_img.shape != (48, 96):
                #     cv2.imshow('', img)
                #     cv2.waitKey(0)
                crop_dict = {'img': crop_img, 'label': label_vertebra[i]}
                vertebra_data[middle_frame.study_uid][vertebra_map[i]] = crop_dict
            for i, coord in enumerate(coord_disc):
                crop_img = img[int(coord[1]) - 24:int(coord[1]) + 24, int(coord[0]) - 48: int(coord[0]) + 48]
                # if crop_img.shape != (48, 96):
                #     cv2.imshow('', img)
                #     cv2.waitKey(0)
                crop_dict = {'img': crop_img, 'label': label_disc[i]}
                disc_data[middle_frame.study_uid][disc_map[i]] = crop_dict
        # vertebra_data = {study_uid: {'T12': {'img': np.array, 'label': tensor}, ...}, ...}
        # disc_data = {study_uid: {'T12-L1': {'img': np.array, 'label': tensor}, ...}, ...}
        return vertebra_data, disc_data


class CoordDataset(data.Dataset):
    def __init__(self, data_path, annotation_path, is_flip=False, is_rot=False, rot_factor=45, is_train=True):
        pkl_path = [config.external_data_path + r'/studies_train.pkl',
                    config.external_data_path + r'/annotation_train.pkl'] \
            if is_train else [config.external_data_path + r'/studies_val.pkl',
                              config.external_data_path + r'/annotation_val.pkl']
        if not (os.path.exists(pkl_path[0]) and os.path.exists(pkl_path[1])):
            self.studies, self.annotation, _ = construct_studies(data_path, annotation_path)
            joblib.dump(self.studies, pkl_path[0])
            joblib.dump(self.annotation, pkl_path[1])
        else:
            self.studies = joblib.load(pkl_path[0])
            self.annotation = joblib.load(pkl_path[1])
        self.middle_list = []
        for study in self.studies.values():
            self.middle_list.append(study.t2_sagittal_middle_frame)
        self.sigma = 1.5
        self.label_type = 'Gaussian'
        self.is_flip = is_flip
        self.is_rot = is_rot
        self.rot_factor = rot_factor
        self.is_train = is_train
        self.mean = np.array([0.485], dtype=np.float32)
        self.std = np.array([0.229], dtype=np.float32)
        self.train_transformation = Compose([
            # HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
            Normalize(),
            ToTensor()
        ])
        self.val_transformation = Compose([
            Normalize(),
            ToTensor()
        ])

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, item):
        middle_frame = self.middle_list[item]
        annotation = self.annotation[middle_frame.study_uid, middle_frame.series_uid, middle_frame.instance_uid]
        coord = torch.cat((annotation[0][:, :2], annotation[1][:, :2]), 0)
        pts = coord.numpy()
        label_vertebra = annotation[0][:, 2]
        label_disc = annotation[1][:, 2]
        img = middle_frame.image
        img = np.asarray(img)
        height, weight = img.shape[0], img.shape[1]
        ori_size = min(height, weight)

        # flip and crop
        if random.random() < 0.5 and self.is_flip:
            img = np.fliplr(img)
            pts = np.array([[weight - i[0], i[1]] for i in pts])
        if weight > height:
            img = img[:, :height]
        elif height > weight:
            img = img[:weight, :]
        else:
            img = img
        input_size = [256, 256]
        img = Image.fromarray(img)

        # rotation
        r = random.uniform(-self.rot_factor, self.rot_factor)
        if random.random() < 0.5 and self.is_rot:
            img = img.rotate(r)
            pts = coordRotate(pts, r, ori_size)
        img = img.resize((input_size[0], input_size[1]))
        tpts = pts * float(input_size[0]) / float(ori_size)
        img = np.asarray(img, dtype=np.uint8)
        img = cv2.merge([img, img, img])
        # show img
        # point_size = 1
        # point_color = (0, 0, 255)  # BGR
        # thickness = 4  # 可以为 0 、4、8
        # for coord in pts:
        #     coord = (int(coord[0]), int(coord[1]))
        #     cv2.circle(img, coord, point_size, point_color, thickness)
        # cv2.imshow('', img)
        # cv2.waitKey(0)

        # normalization img
        # img = (img / 255.0 - self.mean) / self.std
        # img = torch.Tensor(img)
        # img = img.unsqueeze(0)
        if self.is_train:
            img = self.train_transformation(image=img)['image']
        else:
            img = self.val_transformation(image=img)['image']
        n_parts = pts.shape[0]

        # generation heat_map
        target = np.zeros((n_parts, input_size[0], input_size[1]))
        for i in range(n_parts):
            # 逐个坐标点遍历
            if tpts[i, 1] > 0:
                target[i] = generate_target(target[i], tpts[i] - 1, self.sigma,
                                            label_type=self.label_type)
        target = torch.from_numpy(target)
        # meta = {ori size, study uid, series uid, instance uid, path, instance number}
        meta = {'ori': ori_size, 'study_uid': middle_frame.study_uid, 'series_uid': middle_frame.series_uid,
                'instance_uid': middle_frame.instance_uid, 'path': middle_frame.file_path, 'pts': torch.Tensor(pts),
                'ps': middle_frame.pixel_spacing, 'tpts': tpts, 'instance_number': middle_frame.instance_number}

        return img, target, meta


    def collate_fn(self, data):
        imgs, targets, metas = [], [], []
        for img, target, meta in data:
            imgs.append(img)
            targets.append(target)
            metas.append(meta)
        imgs = torch.stack(imgs, dim=0)
        targets = torch.stack(targets, dim=0)
        data = (imgs, targets)
        label = (None, )
        return data, label




class TestDataset(data.Dataset):
    def __init__(self, test_path):
        pkl_path = './test.pkl'
        if not os.path.exists(pkl_path):
            self.test_studies = construct_studies(test_path)
            joblib.dump(self.test_studies, pkl_path)
        else:
            self.test_studies = joblib.load(pkl_path)
        self.test_list = []
        for study in self.test_studies.values():
            self.test_list.append(study.t2_sagittal_middle_frame)
        self.mean = np.array([0.485], dtype=np.float32)
        self.std = np.array([0.229], dtype=np.float32)
        self.transformation = Compose([
            # CLAHE(always_apply=True),
            Normalize(),
            ToTensor()
        ])

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
        img = cv2.merge([img, img, img])
        img = self.transformation(image=img)['image']
        # img = (img / 255.0 - self.mean) / self.std
        # img = torch.Tensor(img)
        # img = img.unsqueeze(0)
        meta = {'ori': ori_size, 'study_uid': middle_frame.study_uid, 'series_uid': middle_frame.series_uid,
                'instance_uid': middle_frame.instance_uid, 'path': middle_frame.file_path,
                'instance_number': middle_frame.instance_number}
        return img, meta, np.array(middle_frame.image)


class TestDatasetB(data.Dataset):
    def __init__(self, test_path, map_json):
        pkl_path = config.external_data_path + '/testB.pkl'
        with open(map_json, 'r', encoding='utf8') as fp:
            series_map = json.load(fp)
        if not os.path.exists(pkl_path):
            self.test_studies = construct_studies(test_path)
            joblib.dump(self.test_studies, pkl_path)
        else:
            self.test_studies = joblib.load(pkl_path)
        # print(self.test_studies)
        self.test_list = []
        for map_dict in series_map:
            try:
                self.test_studies[map_dict['studyUid']].t2_sagittal_uid = map_dict['seriesUid']
            except KeyError:
                continue
            self.test_list.append(self.test_studies[map_dict['studyUid']].t2_sagittal_middle_frame)
            if self.test_studies[map_dict['studyUid']].t2_sagittal_middle_frame.series_uid != map_dict['seriesUid']:
                print('!')
        self.mean = np.array([0.485], dtype=np.float32)
        self.std = np.array([0.229], dtype=np.float32)
        self.transformation = Compose([
            # CLAHE(always_apply=True),
            Normalize(),
            ToTensor()
        ])

    def __len__(self):
        return len(self.test_list)

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
        img = cv2.merge([img, img, img])
        img = self.transformation(image=img)['image']
        # img = (img / 255.0 - self.mean) / self.std
        # img = torch.Tensor(img)
        # img = img.unsqueeze(0)
        meta = {'ori': ori_size, 'study_uid': middle_frame.study_uid, 'series_uid': middle_frame.series_uid,
                'instance_uid': middle_frame.instance_uid, 'path': middle_frame.file_path,
                'instance_number': middle_frame.instance_number}
        return img, meta, np.array(middle_frame.image)


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

        self.all_data_dict, self.all_data_csv = CreatAxialDataset(self.data_root_path, self.data_json_path,
                                                                  is_train=self.is_train)
        # print(self.all_data_dict)
        # print(self.all_data_csv)
        # print(is_train)

        self.transform = transform
        self.totensor = transforms.ToTensor()
        # self.data_root = cfg.DATASET.ROOT
        self.input_size = config.input_size
        self.num_classes = config.num_classes

        self.counter = [0, 0, 0, 0, 0]

        self.map = {'v1': 0, 'v2': 1, 'v3': 2, 'v4': 3, 'v5': 4, 'v1_v': 5, 'v2_v': 6}

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
            print("read dirty data: ", self.img_dir)

        label = self.all_data_dict[idx]['label']  # 获取图片的标签

        # 矢状面的路径
        disc_path = self.all_data_dict[idx]['disc_dcmPath']
        identification = self.all_data_dict[idx]['identification']
        studyID = self.all_data_dict[idx]['studyUid']

        # 有些label为两个值,选择前一个值
        if self.is_train:
            if ',' in label:
                label = label.split(',')[0]
            if identification == 'vertebra':
                label = label + '_v'
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

        if self.is_train:
            return img, label
        else:
            return img, label, disc_path, identification, studyID


class Axial_Testset(data.Dataset):

    def __init__(self, test_dict_path, transform=None):

        self.test_dict_path = test_dict_path
        self.transform = transform
        self.totensor = transforms.ToTensor()

        self.input_size = config.input_size

        self.test_dict = np.load(self.test_dict_path, allow_pickle=True)

    def __len__(self):

        return len(self.test_dict)

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

        return img, axial_path, identification, studyID


class SagAxialDataset(data.Dataset):
    """
    """
    def __init__(self, data_root_path, data_json_path, part, is_train=True, transform=None):
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
        self.part = part
        self.vertebra_list = ['L1', 'L2', 'L3', 'L4', 'L5']
        self.disc_list = ['T12-L1', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']

        # self.csv_file = cfg.DATASET.TESTSET
        # self.data_root = cfg.DATASET.TESTROOT

        self.all_axialdata_dict, self.all_axialdata_csv = CreatAxialDataset(self.data_root_path, self.data_json_path,
                                                                            is_train=self.is_train)

        # self.test_dict = np.load(self.test_dict_path,allow_pickle = True)
        # self.test_csv  = pd.read_csv(self.test_csv_path)

        if is_train:
            sag_train = PrepareCropData(config.trainPath, config.trainjsonPath, "train")
            ##将val注释了 , 目前第一轮还用不到

            # sag_val = PrepareCropData(config.valPath, config.valjsonPath, "val")

            self.sag_total = copy.deepcopy(sag_train)

            # print(type(pre_train.disc_data))

            # for key, study in sag_val.disc_data.items():
            #     self.sag_total.disc_data[key] = study
            #
            # for key, study in sag_val.vertebra_data.items():
            #     self.sag_total.vertebra_data[key] = study

        else:
            self.sag_total = PrepareCropData(config.testPath, config.testjsonPath, "test")

        if self.part == 'vertebra':

            self.part_data_csv = self.all_axialdata_csv.loc[
                                 self.all_axialdata_csv['identification'].isin(self.vertebra_list), :]
            #
            # self.part_data_csv = self.test_csv.loc[self.test_csv['vertebra'] == 1,:]
        elif self.part == 'disc':
            self.part_data_csv = self.all_axialdata_csv.loc[
                                 self.all_axialdata_csv['identification'].isin(self.disc_list), :]
            # self.part_data_csv = self.test_csv.loc[self.test_csv['disc'] == 1,:]
        else:
            print("part value should be  'disc' or 'vertebra' ")
            raise ValueError

        self.part_data_csv.reset_index(drop=True, inplace=True)
        self.part_data_dict = self.part_data_csv.to_dict(orient='records')
        self.map = {'v1': 0, 'v2': 1, 'v3': 2, 'v4': 3, 'v5': 4}
        self.transform = transform
        self.totensor = transforms.ToTensor()
        # self.data_root = cfg.DATASET.ROOT
        self.input_size = config.input_size
        self.num_classes = config.num_classes

        self.counter = [0, 0, 0, 0, 0]

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
            print("read dirty data: ", self.axial_img_dir)

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
            if ',' in label:
                label = label.split(',')[0]
            label = self.map[label]

        axial_height, axial_weight = axial_img.shape[0], axial_img.shape[1]
        sag_height, sag_weight = sag_img.shape[0], sag_img.shape[1]
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
        sag_img = Image.fromarray(sag_img)

        if sag_weight > sag_height:

            self.padding = transforms.Pad(padding=(0, int((sag_weight - sag_height) / 2)))

        else:
            self.padding = transforms.Pad(padding=(int((sag_height - sag_weight) / 2), 0))

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

        return axial_img, sag_img, label, disc_path, identification, studyID

    def collate_fn(self, data):
        axial_imgs, sag_imgs, labels = [], [], []
        for axial_img, sag_img, label, _, _, _ in data:
            axial_imgs.append(axial_img)
            sag_imgs.append(sag_img)
            labels.append(label)
        axial_imgs = torch.stack(axial_imgs, dim=0)
        sag_imgs = torch.stack(sag_imgs, dim=0)
        labels = torch.from_numpy(np.array(labels))
        data = (axial_imgs, sag_imgs)
        return data, labels


class SagAxial_Test_Dataset(data.Dataset):
    """
    """

    def __init__(self, test_dict_path, test_csv_path, part, is_train=True, transform=None):
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
        self.test_csv_path = test_csv_path

        self.is_train = is_train
        self.part = part
        self.vertebra_list = ['L1', 'L2', 'L3', 'L4', 'L5']
        self.disc_list = ['T12-L1', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']

        # self.csv_file = cfg.DATASET.TESTSET
        # self.data_root = cfg.DATASET.TESTROOT

        # self.all_axialdata_dict,self.all_axialdata_csv = CreatAxialDataset(self.data_root_path,self.data_json_path,is_train = self.is_train)
        self.test_dict = np.load(self.test_dict_path, allow_pickle=True)
        self.test_csv = pd.read_csv(self.test_csv_path)

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
            self.sag_total = PrepareCropData(config.testPath, config.testjsonPath, "test")

        if self.part == 'vertebra':

            # self.part_data_csv = self.all_axialdata_csv.loc[self.all_axialdata_csv['identification'].isin(self.vertebra_list), :]

            self.part_data_csv = self.test_csv.loc[self.test_csv['vertebra'] == 1, :]
        elif self.part == 'disc':
            # self.part_data_csv = self.all_axialdata_csv.loc[self.all_axialdata_csv['identification'].isin(self.disc_list), :]
            self.part_data_csv = self.test_csv.loc[self.test_csv['disc'] == 1, :]
        else:
            print("part value should be  'disc' or 'vertebra' ")
            raise ValueError

        self.part_data_csv.reset_index(drop=True, inplace=True)
        self.part_data_dict = self.part_data_csv.to_dict(orient='records')
        self.map = {'v1': 0, 'v2': 1, 'v3': 2, 'v4': 3, 'v5': 4}
        self.transform = transform
        self.totensor = transforms.ToTensor()
        # self.data_root = cfg.DATASET.ROOT
        self.input_size = config.input_size
        self.num_classes = config.num_classes

        self.counter = [0, 0, 0, 0, 0]

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
            print("read dirty data: ", self.axial_img_dir)

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
            if ',' in label:
                label = label.split(',')[0]
            label = self.map[label]

        axial_height, axial_weight = axial_img.shape[0], axial_img.shape[1]
        sag_height, sag_weight = sag_img.shape[0], sag_img.shape[1]
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
        sag_img = Image.fromarray(sag_img)

        if sag_weight > sag_height:

            self.padding = transforms.Pad(padding=(0, int((sag_weight - sag_height) / 2)))

        else:
            self.padding = transforms.Pad(padding=(int((sag_height - sag_weight) / 2), 0))

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

        return axial_img, sag_img, label, disc_path, identification, studyID


if __name__ == '__main__':
    #################  矢状面图像Crop ###################################
    pre_train = PrepareCropData(config.trainPath, config.trainjsonPath, "train")
    pre_val = PrepareCropData(config.valPath, config.valjsonPath, "val")

    # print()

    pre_total = copy.deepcopy(pre_train)

    # print(type(pre_train.disc_data))

    for key, study in pre_val.disc_data.items():
        pre_total.disc_data[key] = study

    for key, study in pre_val.vertebra_data.items():
        pre_total.vertebra_data[key] = study

    img = pre_total.vertebra_data['1.2.1114.250756.6.682.420717188774813080248514492603145951793']['L2']['img']

    plt.figure()
    plt.imshow(img, cmap='gray')

    # sns.distplot(img)
    plt.show()

    print(type(img))

    ##################  矢状面图像Crop ###################################

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
