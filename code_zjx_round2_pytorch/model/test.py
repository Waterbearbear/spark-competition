import torch
from model import model_axial

import config
import os
from datasets import dataset, transform
from backbone.layers import DoubleNet
import numpy as np

import pandas as pd


def test_classification():

    file_path = os.path.dirname(__file__)

    # print("file_path:",file_path)
    parent_path = os.path.dirname(file_path)

    model_name = config.model_name
    net_num    = config.num_classes
    nets_path = os.path.join(parent_path,config.pretrianed_models_dir)



    # test_dataset = dataset_.axialdataset(data_root_path=config.testPath,
    #                                    data_json_path=config.testjsonPath,
    #                                    is_train=False,
    #                                    transform=None)

    # test_dataset = dataset_.Axial_Testset(test_dict_path = config.test_Dict_Path,
    #                                      transform = None)

    # vertebra_test_dataset = dataset_.SagAxialDataset( data_root_path = config.testPath,
    #                                         data_json_path = config.testjsonPath,
    #                                         part = 'vertebra',
    #                                         is_train=config.isTrain,
    #                                         transform= transform.val_transforms())
    #
    # disc_test_dataset = dataset_.SagAxialDataset(data_root_path=config.testPath,
    #                                             data_json_path=config.testjsonPath,
    #                                             part='disc',
    #                                             is_train=config.isTrain,
    #                                             transform=transform.val_transforms())


    ### SagAxial双路 ###
    vertebra_test_dataset = dataset.SagAxial_Test_Dataset(test_dict_path=config.test_Dict_Path,
                                                        test_csv_path=config.test_csv_Path,
                                                        part='vertebra',
                                                        is_train=False,
                                                        transform=transform.val_transforms())
    #
    disc_test_dataset = dataset.SagAxial_Test_Dataset(test_dict_path=config.test_Dict_Path,
                                                    test_csv_path=config.test_csv_Path,
                                                    part='disc',
                                                    is_train=False,
                                                    transform=transform.val_transforms())

    ## 仅Sag一路##


    # vertebra_test_dataset = dataset.CropSagDataset('vertebra', is_train=False, transform=transform.val_transforms())
    # disc_test_dataset = dataset.CropSagDataset('disc', is_train=False, transform=transform.val_transforms())



    ######## 单个模型 ##########
    # model = layers.ResNeSt(modelname = config.model_name,
    #                        pretrained_path = config.pretrainedPath)

    # model_vertebra = layers.DoubleNet(modelname = config.model_name,
    #                                num_classes = config.num_classes[0],
    #                                pretrained_path = None)
    #
    # model_disc  = layers.DoubleNet(modelname = config.model_name,
    #                                num_classes = config.num_classes[1],
    #                                pretrained_path = None)

    # vertebra_model_path = os.path.join(config.checkpoints_dir,"%s_vertebra_fold1_best.pth"%config.model_name)
    # disc_model_path = os.path.join(config.checkpoints_dir,"%s_disc_fold0_best.pth"%config.model_name)
    # disc_model_path = os.path.join(config.checkpoints_dir,"%s_disc_fold1_best.pth"%config.model_name)


    #双路单模型
    # model_vertebra = DoubleNet(modelname= config.model_name, phase= 'vertebra')
    #
    # model_disc = DoubleNet(modelname=config.model_name, phase= 'disc')
    #
    # for phase in ['vertebra','disc']:
    #
    #     parameter_path = os.path.join(nets_path, "%s_%s_fold0_best.pth" % (config.model_name, phase))
    #     # print(parameter_path)
    #
    #     pretrained_dict = torch.load(parameter_path)
    #
    #     if phase == 'vertebra':
    #         model_dict = model_vertebra.state_dict()
    #     elif phase == 'disc':
    #         model_dict = model_disc.state_dict()
    #     else:
    #         raise ValueError
    #
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     model_dict.update(pretrained_dict)
    #
    #     if phase == 'vertebra':
    #         model_vertebra.load_state_dict(model_dict)
    #     elif phase == 'disc':
    #         model_disc.load_state_dict(model_dict)


    #### 双路 Bagging_模型 ########
    model_vertebra = model_axial.Bagging_Double_Model(model_name = config.model_name,
                                              net_num = config.k_fold,
                                              num_classes = config.num_classes[0],
                                              part = 'vertebra',
                                              nets_path = config.models_dir)

    model_disc = model_axial.Bagging_Double_Model(model_name = config.model_name,
                                                  net_num = config.k_fold,
                                                  num_classes = config.num_classes[1],
                                                  part = 'disc',
                                                  nets_path = config.models_dir)



    ## 单路 Bagging模型
    # model_vertebra = model_axial.Bagging_Model(model_name = config.model_name,
    #                                           net_num = config.k_fold,
    #                                           num_classes = config.num_classes[0],
    #                                           part = 'vertebra',
    #                                           nets_path = config.models_dir)
    #
    # model_disc = model_axial.Bagging_Model(model_name = config.model_name,
    #                                               net_num = config.k_fold,
    #                                               num_classes = config.num_classes[1],
    #                                               part = 'disc',
    #                                               nets_path = config.models_dir)


    ## 单路bagging 模型
    # model_vertebra = model_axial.Bagging_Model()

    test_json = pd.read_json(config.testjsonPath)

    for phase in ["vertebra","disc"]:
       if phase == 'vertebra':

           # pretrain_dict = torch.load(vertebra_model_path)
           dataset_ = vertebra_test_dataset
           model   = model_vertebra
       else:
           # pretrain_dict = torch.load(disc_model_path)
           dataset_ = disc_test_dataset
           model = model_disc

       # model_dict = model.state_dict()
       # model_dict.update(pretrain_dict)
       # model.load_state_dict(model_dict)


       test_dataloader = torch.utils.data.DataLoader(
                                                    dataset=dataset_,
                                                    batch_size=config.batch_size,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    )


       test_json = final_test(
                  test_dataloader = test_dataloader,
                  model = model,
                  model_type = config.model_type,
                  test_json = test_json)

    test_json.to_json(config.submision_output_file_path, orient = 'records')


def final_test(test_dataloader,model,model_type,test_json):


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    # map = {'v1': 0, 'v2': 1, 'v3': 2, 'v4': 3, 'v5': 4}
    map = ['v1','v2','v3','v4','v5']

    # test_json = pd.read_json(testjson_path)

    model.to(device)
    model.eval()

    # print('test_dataloader',type(test_dataloader))

    with torch.no_grad():
        for i,(axial_img,sag_img,label,disc_path,identification,studyID) in enumerate(test_dataloader):

            # img = img.repeat([1,3,1,1])

            # print(axial_img.shape)

            if model.model_name == "MobileNet_v2":
                sag_img = sag_img.repeat([1,3,1,1])
                axial_img = axial_img.repeat([1,3,1,1])

            axial_img = axial_img.to(device)
            sag_img = sag_img.to(device)

            if model_type == "Double":
                output = model(sag_img,axial_img)
            elif model_type == "Single":
                output = model(sag_img)

            # print(output)

            class_pred = np.squeeze(output.argmax(dim=1, keepdim=True).cpu().numpy())

            #output.shape = [batch,label]
            pred_one = {}

            for index in range(output.shape[0]):
                disc_path_one = disc_path[index]
                identification_one = identification[index]
                studyID_one = studyID[index]

                data = test_json.loc[test_json['studyUid'] == studyID_one,'data']
                data_dict = data.iloc[0]
                annotation = data_dict[0]['annotation'][0]


                # print(studyID_one)
                for point in annotation['data']['point']:
                    if point['tag']['identification'] == identification_one:
                        if 'disc' in point['tag'].keys():
                            point['tag']['disc'] = map[class_pred[index]]
                        else:
                            point['tag']['vertebra'] = map[class_pred[index]]
                        break

                data_dict[0]['annotation'][0] = annotation
                data.iloc[0] = data_dict
                test_json.loc[test_json['studyUid'] == studyID_one, 'data'] = data


    # print(json_df)
        # 填补null值
        # for idx in test_json.index:
        #
        #     annotation = test_json.loc[idx, "data"][0]['annotation'][0]
        #
        #     for point in annotation['data']['point']:
        #         if 'disc' in point['tag'].keys():
        #             if point['tag']['disc'] == None:
        #                 point['tag']['disc'] = 'v1'
        #
        #         elif 'vertebra' in point['tag'].keys():
        #             if point['tag']['vertebra'] == None:
        #                 point['tag']['vertebra'] = 'v1'
        #
        #
        #     test_json.loc[idx, "data"][0]['annotation'][0] = annotation


        return  test_json
        # test_json.to_json('test_resnest50_kfold_best2.json', orient = 'records')
