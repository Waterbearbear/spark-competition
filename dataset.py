# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image, ImageFile
import numpy as np

from imgread import get_info,dicom2array
from utils import generate_target

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
    def __init__(self, data_root_path,data_json_path, is_train = True, transform=None):
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
        self.input_size = [256,256]
        self.output_size = [64,64]
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


        output_size = [64, 64]
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

        for i in range(len(self.all_data_dict)):
            img_dir = self.all_data_dict[i][0]  # 获取图片的地址
            # print(img_dir)
            img = dicom2array(img_dir)  # 获取具体的图片数据，二维数据
            annotation = self.all_data_dict[i][1]  # 获取图片的标签
            points = annotation[0]['data']['point']
            pts = []
            labels = []
            # print("points: ",points)

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
            img = img.resize((input_size[0], input_size[1]))
            pts *= [output_size[0] / input_size[0], output_size[1] / input_size[1]]

            # print("img.size: ",img.size)

            # scale *= 1.25
            nparts = pts.shape[0]  # Landmark的数量
            print(nparts)
            if nparts != 11:
                print(img_dir)
            # print("nparts: ",nparts)

            target = np.zeros((nparts, output_size[0], output_size[1]))
            ##target 为生成的HeatMap   shape:19 x 64 x 64

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
                                                label_type= self.label_type)
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
            # print(img.shape)

            img = img.astype(np.float32)
            img = (img / 255.0 - self.mean) / self.std
            img = img.transpose([2, 0, 1])
            target = torch.Tensor(target)

            tpts = torch.Tensor(tpts)  # 转化后所有的的坐标值
            # center = torch.Tensor(center) #当前

            # meta = {'index': idx, 'center': center, 'scale': scale,
            #         'pts': torch.Tensor(pts), 'tpts': tpts, 'box_size': box_size}

            meta = {'index': idx, 'pts': torch.Tensor(pts), 'tpts': tpts}

            return img, target, meta


if __name__ == '__main__':
    pass
