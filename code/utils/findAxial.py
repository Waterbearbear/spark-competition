from torchvision import transforms
import torch

# def train_transforms(width, height):
#     trans_list = [
#         transforms.Resize((height, width)),
#         transforms.RandomVerticalFlip(p=0.5),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomApply([
#             transforms.RandomAffine(degrees=20,
#                                     translate=(0.15, 0.15),
#                                     scale=(0.8, 1.2),
#                                     shear=5)], p=0.5),
#         transforms.RandomApply([
#             transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
#         transforms.ToTensor()
#     ]
#     return transforms.Compose(trans_list)

def train_transforms():
    trans_list = [
        # transforms.Normalize(mean = [0.486,0.486,0.486],std = [0.256,0.256,0.256]),
        transforms.ToTensor()
    ]
    return transforms.Compose(trans_list)


def val_transforms():
    trans_list = [
        transforms.ToTensor()
    ]
    return transforms.Compose(trans_list)


def onehot(batch_size,num_classes,label):



    # label = torch.LongTensor(label).view(-1,1)
    gt_label = label.view(-1,1).long()

    # print("label.shape :",gt_label.shape)
    # print("label type: ",type(gt_label))


    # print(label.shape)


    label_onehot = torch.FloatTensor(batch_size,num_classes).cuda()

    # print('one_hot.shape:', label_onehot)
    #
    # print("one_hot.type: ",type(label_onehot))


    # print(label_onehot.shape)

    label_onehot.zero_()
    label_onehot.scatter_(1 , gt_label ,1)

    # print(label)
    # print(label_onehot)

    return label_onehot
