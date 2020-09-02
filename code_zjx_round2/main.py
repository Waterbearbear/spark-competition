from model import train_axial,test
import pandas as pd
import numpy as np
from utils.imgread import CreatAxialDataset,CreatPointToAxialCsv
import config
import copy
import os


from datasets import dataset

# from spinal_code.core.disease.data_loader import DisDataSet
# from spinal_code.core.disease.evaluation import Evaluator
# from spinal_code.core.disease.model import DiseaseModelBase
# from spinal_code.core.key_point import KeyPointModel, NullLoss
# from spinal_code.core.structure import construct_studies
#
# from zoo.common.nncontext import *
# from zoo.pipeline.api.keras.optimizers import Adam
# from zoo.orca.learn.pytorch import Estimator


def MergeDataJson():

    pd.set_option('expand_frame_repr', False)

    # train_json = pd.read_json(config.trainjsonPath)

    train_csv = pd.read_csv(r'..\data\External\axial_info_train.csv')
    val_csv = pd.read_csv(r'..\data\External\axial_info_val.csv')

    # result_test = np.load('result_test.npy')
    # result_train = np.load('result_train.npy')
    frames = [train_csv, val_csv]

    all_csv = pd.concat(frames)

    all_csv.reset_index(drop=True, inplace=True)
    all_csv.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], inplace=True)

    all_csv.to_csv(os.path.join(config.external_data_path,'axial_info_all.csv'))
    all_dict = all_csv.to_dict(orient='records')
    np.save(os.path.join(config.external_data_path,'axial_info_all.npy'), all_dict)


if __name__ == "__main__":

    ###################分类################################

    ###### 所有训练设置都在config中完成
    test_result_dict_Path = os.path.join(config.external_data_path, 'result_test.npy')
    test_all_axial_Path = os.path.join(config.external_data_path, 'dcm_info_test.csv')

    ###创建train 150例和val 51例文件的csv和dict 用于dataset
    # CreatAxialDataset(dicomPath=config.trainPath, jsonPath=config.trainjsonPath, is_train=True)
    # CreatAxialDataset(dicomPath=config.valPath, jsonPath=config.valjsonPath, is_train=True)

    ### 将train和val合并,做5折交叉验证
    # MergeDataJson()

    ### 训练开始  ########

    train_axial.train_axial_model()

    ### 创建测试训练用的csv和dict
    # CreatAxialDataset(dicomPath=config.testPath, jsonPath=config.testjsonPath, is_train=False)

    ### 找出预测点对应的轴状图,保存成dict和csv
    # csv, dict = CreatPointToAxialCsv(result_dict_path=test_result_dict_Path,
    #                                  all_axial_csv_path=test_all_axial_Path,
    #                                  is_train=False)

    ## 进行测试
    ## 测试需要的用的
    test.test_classification()

    ###########分类##################


    ##########测试#############

    # print("total vertebra")
    # sag_total =


    # for key,study in sag_total.vertebra_data.items():
    #     print(sag_total.vertebra_data)


    # pass
    # print(pre_val)
