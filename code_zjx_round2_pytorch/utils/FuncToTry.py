
import config

# from utils import generate_target
from datasets import dataset , transform
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from resnest.torch import resnest18
import torch

import torch.nn as nn
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


    valPath = r'E:\BME\competition\spark\data\lumbar_train51'
    valjsonPath = r'E:\BME\competition\spark\data\lumbar_train51_annotation.json'

    trainPath = r'E:\BME\competition\spark\data\lumbar_train150'
    trainjsonPath = r'E:\BME\competition\spark\data\lumbar_train150_annotation.json'

    testPath = r"E:\BME\competition\spark\data\lumbar_testA50"
    testjsonPath = r"E:\BME\competition\spark\data\test1.json"

    pd.set_option('expand_frame_repr', False)

    net = resnest18(pretrained = False)

    print(net)
    net.conv1 = nn.Sequential(
                nn.Conv2d(1,32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            )



    net.fc = nn.Linear(in_features = 2048,
                       out_features = 5,
                       bias = True)

    print(net)

    input = torch.randn([32,1,224,224])
    output = net(input)
    print(input)
    print(output)

    # train_json = pd.read_json(config.trainjsonPath)
    #
    # train_csv = pd.read_csv('axial_info_train.csv')
    # val_csv = pd.read_csv('axial_info_val.csv')
    #
    # result_test = np.load('result_test.npy')
    # result_train = np.load('result_train.npy')
    #
    #
    # frames = [train_csv,val_csv]
    #
    #
    # test_studyUid = "1.3.6.1.4.1.43960.1.1.10503147.60120337.6399"
    #
    # # print(test_json)
    #
    # test_json = pd.read_json(config.testjsonPath,orient= 'records')
    #
    #
    # print(test_json)
    # print(test_json.loc[test_json['studyUid'] == test_studyUid,:])
    #
    # data = test_json.loc[test_json['studyUid'] == test_studyUid,'data']
    #
    # # print(data.iloc[0])
    # data_dict = data.iloc[0]
    # # print(data_dict[0])
    #
    #
    #
    # # print(data_dict[0]['annotation'][0])
    #
    # annotation = data_dict[0]['annotation'][0]
    # print("annotation: ",annotation['data']['point'])
    #
    #
    #
    # for point in annotation['data']['point']:
    #
    #
    #     print("point: ",point)
    #
    #     point['tag']['disc'] = 'v1'
    #
    # data_dict[0]['annotation'][0] = annotation
    # data.iloc[0] = data_dict
    #
    # print(data.iloc[0])
    #
    # test_json.loc[test_json['studyUid'] == test_studyUid, 'data'].iloc[0] = data.iloc[0]
    #
    # print(test_json.loc[test_json['studyUid'] == test_studyUid, 'data'].iloc[0])


    # print(data[0])

    # all_csv = pd.concat(frames)
    #
    # all_csv.reset_index(drop=True, inplace=True)
    # all_csv.drop(columns = ['Unnamed: 0','Unnamed: 0.1'], inplace = True)
    #
    #
    #
    # all_csv.to_csv('all.csv')
    # all_dict = all_csv.to_dict(orient= 'records')

    # print(type(all_dict[0]))
    #
    # v1_data = all_csv.loc[all_csv['label'] == 'v1']
    # v2_data = all_csv.loc[all_csv['label'] == 'v2']
    # v3_data = all_csv.loc[all_csv['label'] == 'v3']
    # v4_data = all_csv.loc[all_csv['label'] == 'v4']
    # v5_data = all_csv.loc[all_csv['label'] == 'v5']
    #
    #
    # print(v5_data.iloc[0,:])
    # print(len(v5_data) + len(v4_data) + len(v3_data) + len(v2_data) + len(v1_data))

    # print(all_csv.loc[all_csv['label'] == 'v1'])



    # annotation = result_test[0]['annotation']
    # annotation_trian = result_train[0]['annotation']
    #
    # print(annotation[0]['point'])
    # print(annotation_trian[0]['data']['point'])
    # print(annotation[0])
    # print(annotation[0]['data'])
    #
    # points = annotation[0]['data']['point']
    #
    # print(points)


    # print(train_json)
    # print(test_json)

    # print(test_json[0])
    # print(type(test_json))
    # print(test_json[0])

    #
    # train_dataset = dataset.sparkset(data_root_path = trainPath ,
    #                                  data_json_path = trainjsonPath,
    #                                  is_train = True,
    #                                  transform= transform.train_transforms())
    #
    # val_dataset = dataset.sparkset(data_root_path=valPath,
    #                                  data_json_path=valjsonPath,
    #                                  is_train=True,
    #                                  transform=transform.train_transforms())
    #
    #
    # train_data_loader = DataLoader(dataset= train_dataset ,
    #                                batch_size= config.batch_size,
    #                                shuffle = True,
    #                                num_workers = 0)
    #
    # val_data_loader = DataLoader(dataset=val_dataset,
    #                                batch_size=config.batch_size,
    #                                shuffle=True,
    #                                num_workers=0)
    #
    # axial_train_dataset = dataset.axialdataset(data_root_path=trainPath,
    #                                  data_json_path=trainjsonPath,
    #                                  is_train=True,
    #                                  transform=transform.train_transforms())
    #
    # axial_val_dataset = dataset.axialdataset(data_root_path=valPath,
    #                                            data_json_path=valjsonPath,
    #                                            is_train=True,
    #                                            transform=transform.train_transforms())
    #
    #
    # axial_train_data_loader = DataLoader(dataset=axial_train_dataset,
    #                                batch_size= config.batch_size,
    #                                shuffle=True,
    #                                num_workers=0)
    #
    # axial_val_data_loader = DataLoader(dataset=axial_val_dataset,
    #                                      batch_size=config.batch_size,
    #                                      shuffle=True,
    #                                      num_workers=0)
    #
    #
    # for i,(img, target, meta) in enumerate(train_data_loader):
    #     print("img value",img[img > 0.0])
    #
    #
    #     print("img.shape: ",img.shape)
    #     print("target.shape: ",target.shape)
    #     # print("meta: ",meta)
    #
    #
    # for i,(img,label) in enumerate(axial_train_data_loader):
    #
    #     batch_size = img.shape[0]
    #
    #
    #
    #     label = transform.onehot(batch_size,config.num_classes,label)
    #
    # for i, (img, target, meta) in enumerate(val_data_loader):
    #     print("img value", img[img > 0.0])
    #
    #     print("img.shape: ", img.shape)
    #     print("target.shape: ", target.shape)
    #     # print("meta: ",meta)
    #
    # for i, (img, label) in enumerate(axial_val_data_loader):
    #     batch_size = img.shape[0]
    #
    #     label = transform.onehot(batch_size, config.num_classes, label)
    #
    #



    # result = get_info(trainPath, trainjsonPath)  #获取图片路径及对应的annotation
    # result = get_info(valiPath, valijsonPath)
    # result[:][:] = result
    # result = np.squeeze(result)
    # print(result[0])
    # print(result[0][1][0]['data']['point'][0]['tag'])
    # print(result[0][0])
    # print(len(result))
    # print(type(result))
    # print(type(result))
    # print(result)


    # shape_list = []
    #
    # # img = cv2.imread(r"E:\BME\HRNet-Facial-Landmark-Detection-master\CEP\images\TrainData\051.bmp")
    #
    # # print(type(img))
    # # print(img.shape)
    #
    # for i in range(len(result)):
    #     img_dir = result[i][0]  # 获取图片的地址
    #     # print(img_dir)
    #     img = dicom2array(img_dir)  # 获取具体的图片数据，二维数据
    #     annotation = result[i][1]  # 获取图片的标签
    #     points = annotation[0]['data']['point']
    #     pts = []
    #     labels = []
    #     # print("points: ",points)
    #
    #     for point in points:
    #         # print("point",point)
    #         coord = point['coord']
    #         tag = point["tag"]
    #         # center_x, center_y =
    #         coord_xy = [coord[0], coord[1]]
    #         pts.append(coord_xy)
    #
    #         if "disc" in tag.keys():
    #             # print("tag: ",tag)
    #             label = tag['disc']
    #
    #         if "vertebra" in tag.keys():
    #             label = tag["vertebra"]
    #
    #
    #         # print("label： ",label)
    #         labels.append(label)
    #         # identification = tag["tag"]['identification']
    #
    #         # print(identification)
    #         # print(tag)
    #     pts = np.array(pts)
    #     pts = pts.astype('float').reshape(-1, 2)
    #     # print("pts.shape: ",pts.shape)
    #     # print("labels: ",labels)
    #     # print(type(img))
    #     # img = img
    #     #坐标也要随之变化
    #
    #     height,weight = img.shape[0],img.shape[1]
    #
    #     # print(img.shape)
    #     # print("before: ", img.shape)
    #     if weight > height:
    #
    #         extra_weight = weight - height
    #         crop_left = int(extra_weight/2) - 1
    #         crop_right = int(extra_weight/2) + height -1
    #
    #         pts = pts - [0,crop_left]
    #         # 必须要用逗号索引
    #         img = img[:,crop_left:crop_right]
    #
    #
    #     elif height > weight:
    #         extra_height = height - weight
    #         crop_up = int(extra_height/2) - 1
    #         crop_down = int(extra_height/2) + weight - 1
    #
    #         pts = pts - [crop_up,0]
    #
    #         img = img[crop_up:crop_down,:]
    #
    #
    #
    #     img = Image.fromarray(img)
    #     img = img.resize((input_size[0], input_size[1]))
    #     pts *= [output_size[0]/input_size[0],output_size[1]/input_size[1]]
    #
    #
    #
    #     # print("img.size: ",img.size)
    #
    #     # scale *= 1.25
    #     nparts = pts.shape[0]  # Landmark的数量
    #     print(nparts)
    #     if nparts != 11:
    #         print(img_dir)
    #     # print("nparts: ",nparts)
    #
    #     target = np.zeros((nparts, output_size[0], output_size[1]))
    #     ##target 为生成的HeatMap   shape:19 x 64 x 64
    #
    #     # 自己添加的坐标变换，读入的明明是原图的坐标 但是generate heatmap的时候却用了64x64下的坐标
    #     # 是在transform_pixel的时候处理了
    #     tpts = pts.copy() * [float(output_size[0]) / float(input_size[0]),
    #                          float(output_size[0]) / float(input_size[0])]
    #
    #     ##复制一份Landmark坐标点
    #     ##transformed points
    #
    #     for i in range(nparts):
    #         # 逐个坐标点遍历
    #         if tpts[i, 1] > 0:
    #             # 如果y坐标>0 ?
    #             # 将Landmark进行对应的角度变化和 scale的变化 ?
    #             # tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
    #             #                                scale, self.output_size, rot=r)
    #
    #             # tpts[i, 0:2] = transform_pixel(tpts[i, 0:2] + 1,
    #             #                                self.output_size)
    #
    #             # 生成heatmap图 target[i] 表示第i个点的Heatmap
    #             # sigma为超参数
    #             # 在这部分可能有精度的损失,每次产生的Heatmap不同
    #             target[i] = generate_target(target[i], tpts[i] - 1, sigma,
    #                                         label_type='Gaussian')
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


    # print(shape_list)
    #
    #
    # temp = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    # print(temp.shape)
    #
    # for i in range(4):
    #     for j in range(3):
    #         # print(temp[i][j])
    #         pass