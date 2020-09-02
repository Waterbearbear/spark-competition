
import config

# from utils import generate_target
from datasets import dataset , transform
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from resnest.torch import resnest18
import torch

import torch.nn as nn
from utils.imgread import CreatCropSagDataset
from datasets.dataset import PrepareCropData
import copy

def second(lt):

    print(lt)
    max = 0

    s = {}
    for i in range(len(lt)):

        flag = 0

        for j in range(len(lt)):

            if lt[i] <= lt[j] and i != j:

                flag = flag + 1
        s[i] = flag
        if flag > max:

            max = flag
    # print(s)
    for i in s:

        if s[i] == max - 1:

            break

    return i


if __name__ == '__main__':


    # valPath = r'E:\BME\competition\spark\data\lumbar_train51'
    # valjsonPath = r'E:\BME\competition\spark\data\lumbar_train51_annotation.json'
    #
    # trainPath = r'E:\BME\competition\spark\data\lumbar_train150'
    # trainjsonPath = r'E:\BME\competition\spark\data\lumbar_train150_annotation.json'
    #
    # testPath = r"E:\BME\competition\spark\data\lumbar_testA50"
    # testjsonPath = r"E:\BME\competition\spark\data\test1.json"

    pd.set_option('expand_frame_repr', False)

    sag_train = dataset.PrepareCropData(config.trainPath, config.trainjsonPath, "train")
    sag_val = dataset.PrepareCropData(config.valPath, config.valjsonPath, "val")

    sag_total = copy.deepcopy(sag_train)

    # print(type(pre_train.disc_data))

    ##150和51的数据合成为一个
    for key, study in sag_val.disc_data.items():
        sag_total.disc_data[key] = study

    for key, study in sag_val.vertebra_data.items():
        sag_total.vertebra_data[key] = study


    study = sag_total.disc_data['1.3.6.1.4.1.43960.1.1.10363147.60120337.8794']
    img = study['T12-L1']['img']


    # print()
    print("img: ",img)
    print("img.shape: ",img.shape)

    # sag_disc = CreatCropSagDataset(sag_total,'disc')
    # sag_vertebra = CreatCropSagDataset(sag_total,'vertebra')
    #
    # for i in range(len(sag_disc)):
    #     print(sag_disc[i]['img'])
    #     print(sag_disc[i]['label'])
    #     print(type(sag_disc[i]['img']))
    #     print(type(sag_disc[i]['label']))