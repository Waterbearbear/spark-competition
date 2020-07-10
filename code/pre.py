import argparse

import torch
import cv2
import numpy as np
from model import Model
from structure import construct_studies
from utils.imgread import get_info, dicom2array
from dataset import TestDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default=r'F:/DATA/Lumbar/lumbar_testA50/')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoint')
    parser.add_argument('--isTrain', type=bool, default=False)
    parser.add_argument('--continue_train', type=bool, default=False)
    parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
    parser.add_argument('--weights', type=int, default=249)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    opt = parser.parse_args()
    testdataset = TestDataset('F:/DATA/Lumbar/lumbar_testA50/')
    model = Model(opt)
    dataloader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=1, shuffle=False)
    test_list = []
    for i, data in enumerate(dataloader):
        meta = data[1]
        print(meta)
        ori_img = dicom2array(meta['path'][0])
        print(ori_img)
        model.set_pre_input(data[0], data[1])
        model.test()
        _, preCoord = model.getPreCoord()
        point_list = []
        for j, coord in enumerate(preCoord):
            point = {'coord': [coord[0], coord[1]],}
        # point_size = 1
        # point_color = (0, 0, 255)  # BGR
        # thickness = 4  # 可以为 0 、4、8
        # for coord in preCoord:
        #     coord = (int(coord[0]), int(coord[1]))
        #     cv2.circle(ori_img, coord, point_size, point_color, thickness)
        # cv2.imshow('', ori_img)
        # cv2.waitKey(0)





