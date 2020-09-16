from jizhu.model import train_axial, test
import pandas as pd
import numpy as np
from jizhu.utils.imgread import CreatAxialDataset, CreatPointToAxialCsv
import jizhu.config as config
import json
import os
import argparse
import torch.utils.data
from jizhu.backbone.net import UNet_3Plus
from jizhu.model.coordmodel import SparkModel, NullLoss
from jizhu.model.coordmodel_spark import CoordModel
from jizhu.model.coordtrain import train
from jizhu.utils.imgread import dicom2array
from jizhu.datasets.dataset import CoordDataset, TestDataset

# from spinal_code.core.disease.data_loader import DisDataSet
# from spinal_code.core.disease.evaluation import Evaluator
# from spinal_code.core.disease.model import DiseaseModelBase
# from spinal_code.core.key_point import KeyPointModel, NullLoss
# from spinal_code.core.structure import construct_studies

from zoo.common.nncontext import *
from zoo.pipeline.api.keras.optimizers import Adam
from zoo.orca.learn.pytorch import Estimator

ide = ['T12-L1', 'L1', 'L1-L2',
       'L2', 'L2-L3', 'L3',
       'L3-L4', 'L4', 'L4-L5',
       'L5', 'L5-S1']


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

    all_csv.to_csv(os.path.join(config.external_data_path, 'axial_info_all.csv'))
    all_dict = all_csv.to_dict(orient='records')
    np.save(os.path.join(config.external_data_path, 'axial_info_all.npy'), all_dict)


if __name__ == "__main__":

    # parameter setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default=config.trainPath)
    parser.add_argument('--train_json', type=str, default=config.trainjsonPath)
    # parser.add_argument('--val_path', type=str, default=config.valPath)
    # parser.add_argument('--val_json', type=str, default=config.valjsonPath)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoint')
    parser.add_argument('--isTrain', type=bool, default=True)
    parser.add_argument('--continue_train', type=bool, default=False)
    parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
    parser.add_argument('--epoch', type=int, default=10, help='epoch')
    parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--weights', type=int, default=0)
    parser.add_argument('--num_workers', '-n', type=int, default=4,
                        help="The number of Horovod workers launched for distributed training.")
    parser.add_argument('--worker_cores', '-c', type=int, default=4,
                        help='The number of cores allocated for each worker.')
    parser.add_argument('--master', '-m', type=str,
                        help='The Spark master address of a standalone cluster if any.')
    parser.add_argument('--use_bf16', type=bool, default=True,
                        help='Whether to use BF16 for model training if you are running on a server with BF16 support.')

    opt = parser.parse_args()

    sc = init_spark_standalone(
        master=opt.master,
        num_executors=opt.num_workers,
        executor_cores=opt.worker_cores,
        executor_memory="30g",
        driver_memory="40g",
        conf={"spark.executorEnv.LD_LIBRARY_PATH": "/opt/conda/lib:/opt/conda/envs/tianchi_zoo/lib/:"})


    # predict coord training start

    # val_dataset = CoordDataset(opt.val_path, opt.val_json, is_flip=False, is_rot=False, is_train=False)
    def train_dataloader():
        train_dataset = CoordDataset(opt.train_path, opt.train_json, is_flip=True, is_rot=True, is_train=True)
        return torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=3,
                                           pin_memory=False, collate_fn=train_dataset.collate_fn)


    # val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=opt.batchsize, shuffle=True)
    # unet_model = UNet_3Plus(in_channels=3, n_classes=11)
    # coord_model = CoordModel(model=unet_model)
    # train(opt, train_dataloader, val_dataloader, model)
    # # predict coord training end
    # estimator = Estimator.from_torch(model=coord_model, loss=NullLoss(),
    #                                  optimizer=Adam(lr=1e-4), backend="bigdl")
    # estimator.fit(data=train_dataloader, epochs=opt.epoch)
    # trained_model = estimator.get_model()
    # # predict coord testing start
    # trained_model.eval()
    # testdataset = TestDataset(config.testPath)
    # opt.isTrain = False
    # dataloader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=1, shuffle=False)
    # json_list = []
    # for i, data in enumerate(dataloader):
    #     meta = data[1]
    #     ori_img = dicom2array(meta['path'][0])
    #     _, preCoord = trained_model((data[0], data[1]))
    #     vertebra_coord = preCoord[:5]
    #     disc_coord = preCoord[5:]
    #     preCoord = []
    #     j, k = 0, 0
    #     for i in range(11):
    #         if i % 2 == 0:
    #             preCoord.append(disc_coord[j])
    #             j += 1
    #         else:
    #             preCoord.append(vertebra_coord[k])
    #             k += 1
    #     annotation = {"annotator": 13, "data": {"point": []}}
    #     for j, coord in enumerate(preCoord):
    #         point = {"coord": [np.around(coord[0]), np.around(coord[1])],
    #                  'tag': {'identification': ide[j], 'disc' if j % 2 == 0 else 'vertebra': None},
    #                  'zIndex': int(meta['instance_number'][0]) - 1}
    #         annotation['data']["point"].append(point)
    #     data_list = {"instanceUid": meta['instance_uid'][0], "seriesUid": meta['series_uid'][0],
    #                  "annotation": [annotation]}
    #     test_list = {"studyUid": meta['study_uid'][0], "data": [data_list]}
    #     # print(meta['study_uid'][0])
    #     json_list.append(test_list)
    #     # point_size = 1
    #     # point_color = (0, 0, 255)  # BGR
    #     # thickness = 4  # 可以为 0 、4、8
    #     # ori_img = cv2.merge([ori_img, ori_img, ori_img])
    #     # for coord in preCoord:
    #     #     coord = (int(coord[0]), int(coord[1]))
    #     #     cv2.circle(ori_img, coord, point_size, point_color, thickness)
    #     # cv2.imshow('', ori_img)
    #     # cv2.waitKey(0)
    # print(json_list)
    # jsondata = json.dumps(json_list)
    # f = open(config.testjsonPath, 'w')
    # f.write(jsondata)
    # f.close()
    # # predict coord testing end

    ###################分类#################################

    ###### 所有训练设置都在config中完成
    test_result_dict_Path = os.path.join(config.external_data_path, 'result_test.npy')
    test_all_axial_Path = os.path.join(config.external_data_path, 'dcm_info_test.csv')

    ###创建train 150例和val 51例文件的csv和dict 用于dataset
    ## 创建round2的训练用例,用训练用力作五折交叉验证

    CreatAxialDataset(dicomPath=config.trainPath, jsonPath=config.trainjsonPath, is_train=True)
    # CreatAxialDataset(dicomPath=config.valPath, jsonPath=config.valjsonPath, is_train=True)

    ### 将train和val合并,做5折交叉验证
    ### 这个是将初赛的train150 和 train51 进行合并
    ### TO DO :

    ## 该方法日后实现, 将round1 的train val 和 round2的train 融合
    # MergeDataJson()

    ###  用AZ框架进行训练  训练开始  ########
    ## trained_model_dict含有两个模型
    ## trained_model_dict['vertebra'] 和 trained_model_dict['disc']

    trained_model_dict = train_axial.TrainAxialModelinSpark()

    ### 创建测试训练用的csv和dict
    CreatAxialDataset(dicomPath=config.testPath, jsonPath=config.testjsonPath, is_train=False)

    ### 找出预测点对应的轴状图,保存成dict和csv
    csv, dict = CreatPointToAxialCsv(result_dict_path=test_result_dict_Path,
                                     all_axial_csv_path=test_all_axial_Path,
                                     is_train=False)

    ## 进行测试
    ## 测试需要的用Keypoint定位后的json文件

    test.test_classification()

    sc.stop()
    if not opt.master:
        stop_spark_standalone()

    ###########分类##################
