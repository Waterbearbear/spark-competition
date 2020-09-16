import torch
from ..model import model_axial
from ..backbone import layers
import jizhu.config as config
import os
from ..datasets import dataset, transform
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report
from ..utils import util
import torch.nn as nn
import datetime
import pandas as pd


def final_test(test_dataloader, model, testjson_path, test_json):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # map = {'v1': 0, 'v2': 1, 'v3': 2, 'v4': 3, 'v5': 4}
    map = ['v1', 'v2', 'v3', 'v4', 'v5']

    # test_json = pd.read_json(testjson_path)

    model.to(device)
    model
    model.eval()

    # print('test_dataloader',type(test_dataloader))

    with torch.no_grad():
        for i, (axial_img, sag_img, label, disc_path, identification, studyID) in enumerate(test_dataloader):

            # img = img.repeat([1,3,1,1])

            axial_img = axial_img.to(device)
            sag_img = sag_img.to(device)

            output = model(sag_img, axial_img)

            print(output)

            class_pred = np.squeeze(output.argmax(dim=1, keepdim=True).cpu().numpy())

            # output.shape = [batch,label]
            pred_one = {}

            for index in range(output.shape[0]):
                disc_path_one = disc_path[index]
                identification_one = identification[index]
                studyID_one = studyID[index]

                data = test_json.loc[test_json['studyUid'] == studyID_one, 'data']
                data_dict = data.iloc[0]
                annotation = data_dict[0]['annotation'][0]

                print(studyID_one)
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

        return test_json
        # test_json.to_json('test_resnest50_kfold_best2.json', orient = 'records')


def final_test2(test_dataloader, model, testjson_path):
    pass


# def Bagging_validate(dataloader,model):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     model
#     model.eval()
#
#     test_loss = 0
#     correct = 0
#     results = []
#
#     criteria = nn.CrossEntropyLoss()
#     # criteria = model_axial.focal_loss(alpha=[0.1, 0.2, 0.2, 0.5, 0.5], gamma=2, num_classes=5)
#     # Don't update model
#     with torch.no_grad():
#         tpr_list = []
#         fpr_list = []
#         predlist = []
#         scorelist = []
#         targetlist = []
#         # Predict
#         for batch_index, (img, label) in enumerate(dataloader):
#             data, target = img.to(device), label.to(device)
#
#             step_batch = data.shape[0]
#
#             output = model(data)
#
#             # all_pred =
#             # print("output:")
#             #
#             # print("output: ",output)
#             #  output = {"netv1": [batch,class_num],"netv2":[batch,class_num]......}
#
#             # print("output.shape: ,",output.shape)
#
#             all_pred = torch.zeros([net_num, step_batch, config.num_classes])
#
#             for i, key in enumerate(output):
#                 all_pred[i] = output[key]
#
#             all_pred = all_pred.cpu().numpy()
#             temp1 = np.argmax(all_pred, axis=2)
#
#             pred_result = np.zeros([step_batch, 1])
#             for batch in range(step_batch):
#                 temp2 = np.bincount(temp1[:, batch].reshape(-1))
#                 pred_result[batch] = np.argmax(temp2)
#
#             # pred_result = np.argmax(np.bincount(np.argmax(all_pred,axis = 2)))
#
#             # score = F.softmax(output, dim=1)
#             # pred = output.argmax(dim=1, keepdim=True)
#
#             # print("pred:",pred)
#             # print("pred.shape: ",pred.shape)
#             # shape =
#
#             #             print('target',target.long()[:, 2].view_as(pred))
#             # correct += pred.eq(target.long().view_as(pred)).sum().item()
#
#             #             print(output[:,1].cpu().numpy())
#             #             print((output[:,1]+output[:,0]).cpu().numpy())
#             #             predcpu=(output[:,1].cpu().numpy())/((output[:,1]+output[:,0]).cpu().numpy())
#             targetcpu = target.long().cpu().numpy()
#
#             predlist = np.append(predlist, pred_result)
#             # scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
#             targetlist = np.append(targetlist, targetcpu)
#
#     acc, f1, prec, rec = util.clf_metrics(predictions=predlist,
#                                           targets=targetlist,
#                                           average="macro")
#
#     report = classification_report(targetlist, predlist, output_dict=True)
#
#     # print(report)
#
#     f1_dict = {}
#     map_dict = {'v1': '0.0',
#                 'v2': '1.0',
#                 'v3': '2.0',
#                 'v4': '3.0',
#                 'v5': '4.0'
#                 }
#     for key in config.target_names:
#         items = report[map_dict[key]]
#
#         print(items)
#         f1_dict[key] = items['f1-score']
#
#     print("test | Accuracy {:.4f} | F1 {:.4f} | Precision {:.4f} | "
#           "Recall {:.4f} | Test loss{:.4f}".format(acc, f1, prec, rec, test_loss))
#
#     print("test end")

# if __name__ == "__main__":

def test_classification():
    file_path = os.path.dirname(__file__)

    # print("file_path:",file_path)
    parent_path = os.path.dirname(file_path)

    model_name = config.model_name
    net_num = config.num_classes
    nets_path = os.path.join(parent_path, config.models_dir)

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

    vertebra_test_dataset = dataset.SagAxial_Test_Dataset(test_dict_path=config.test_Dict_Path,
                                                          test_csv_path=config.test_csv_Path,
                                                          part='vertebra',
                                                          is_train=False,
                                                          transform=transform.val_transforms())

    disc_test_dataset = dataset.SagAxial_Test_Dataset(test_dict_path=config.test_Dict_Path,
                                                      test_csv_path=config.test_csv_Path,
                                                      part='disc',
                                                      is_train=False,
                                                      transform=transform.val_transforms())

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

    ####  Bagging_模型 ########
    model_vertebra = model_axial.Bagging_Double_Model(model_name=config.model_name,
                                                      net_num=config.k_fold,
                                                      num_classes=config.num_classes[0],
                                                      part='vertebra',
                                                      nets_path=config.models_dir)

    model_disc = model_axial.Bagging_Double_Model(model_name=config.model_name,
                                                  net_num=config.k_fold,
                                                  num_classes=config.num_classes[1],
                                                  part='disc',
                                                  nets_path=config.models_dir)

    test_json = pd.read_json(config.testjsonPath)

    for phase in ["vertebra", "disc"]:
        if phase == 'vertebra':

            # pretrain_dict = torch.load(vertebra_model_path)
            dataset_ = vertebra_test_dataset
            model = model_vertebra
        else:
            # pretrain_dict = torch.load(disc_model_path)
            dataset_ = disc_test_dataset
            model = model_disc

        # model_dict = model.state_dict()
        # model_dict.update(pretrain_dict)
        # model.load_state_dict(model_dict)

        test_dataloader = torch.utils.data.DataLoader(dataset=dataset_,
                                                      batch_size=config.batch_size,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      )

        test_json = final_test(test_dataloader=test_dataloader,
                               model=model,
                               testjson_path=config.testjsonPath,
                               test_json=test_json)

    test_json.to_json("../submit/submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".json",
                      orient='records')
