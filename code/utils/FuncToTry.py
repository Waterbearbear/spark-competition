
import config

# from utils import generate_target
from datasets import dataset , transform
from torch.utils.data import DataLoader


import torch


if __name__ == '__main__':


    valPath = r'E:\BME\competition\spark\data\lumbar_train51'
    valjsonPath = r'E:\BME\competition\spark\data\lumbar_train51_annotation.json'

    trainPath = r'E:\BME\competition\spark\data\lumbar_train150'
    trainjsonPath = r'E:\BME\competition\spark\data\lumbar_train150_annotation.json'




    train_dataset = dataset.sparkset(data_root_path = trainPath ,
                                     data_json_path = trainjsonPath,
                                     is_train = True,
                                     transform= transform.train_transforms())

    val_dataset = dataset.sparkset(data_root_path=valPath,
                                     data_json_path=valjsonPath,
                                     is_train=True,
                                     transform=transform.train_transforms())


    train_data_loader = DataLoader(dataset= train_dataset ,
                                   batch_size= config.batch_size,
                                   shuffle = True,
                                   num_workers = 0)

    val_data_loader = DataLoader(dataset=val_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=0)

    axial_train_dataset = dataset.axialdataset(data_root_path=trainPath,
                                     data_json_path=trainjsonPath,
                                     is_train=True,
                                     transform=transform.train_transforms())

    axial_val_dataset = dataset.axialdataset(data_root_path=valPath,
                                               data_json_path=valjsonPath,
                                               is_train=True,
                                               transform=transform.train_transforms())


    axial_train_data_loader = DataLoader(dataset=axial_train_dataset,
                                   batch_size= config.batch_size,
                                   shuffle=True,
                                   num_workers=0)

    axial_val_data_loader = DataLoader(dataset=axial_val_dataset,
                                         batch_size=config.batch_size,
                                         shuffle=True,
                                         num_workers=0)


    for i,(img, target, meta) in enumerate(train_data_loader):
        print("img value",img[img > 0.0])


        print("img.shape: ",img.shape)
        print("target.shape: ",target.shape)
        # print("meta: ",meta)


    for i,(img,label) in enumerate(axial_train_data_loader):

        batch_size = img.shape[0]



        label = transform.onehot(batch_size,config.num_classes,label)

    for i, (img, target, meta) in enumerate(val_data_loader):
        print("img value", img[img > 0.0])

        print("img.shape: ", img.shape)
        print("target.shape: ", target.shape)
        # print("meta: ",meta)

    for i, (img, label) in enumerate(axial_val_data_loader):
        batch_size = img.shape[0]

        label = transform.onehot(batch_size, config.num_classes, label)





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
