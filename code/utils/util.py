import sys

sys.path.append(r'.\utils')

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, \
    accuracy_score
from utils.imgread import dicom_metainfo, dicom2array


def clf_metrics(predictions, targets, average='macro'):
    f1 = f1_score(targets, predictions, average=average)
    precision = precision_score(targets, predictions, average=average, zero_division=0)
    recall = recall_score(targets, predictions, average=average, zero_division=0)
    acc = accuracy_score(targets, predictions)

    return acc, f1, precision, recall


def generate_target(img, pt, sigma, label_type='Gaussian'):
    # img: 19 x 64 x 64 的初始化heatmap图
    # pt : landmark的坐标 -1  这个landmark指的是在heatmap  size X size 下的坐标
    # sigma: 生成heatmap 的超参数
    # label_type: 生成Heatmap的方式?

    # Check that any part of the gaussian is in-bounds
    tmp_size = sigma * 3
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]  # 左上坐标 upleft
    br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]  # 右下角坐标 bottomright

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if label_type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    else:
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def coordRotate(pts, r, size):
    center = size / 2
    newpts = []
    for pt in pts:
        pt = pt - center
        rad = r * np.pi / 180
        newpts.append([int(pt[0] * np.cos(rad) + pt[1] * np.sin(rad) + center),
                       int(pt[1] * np.cos(rad) - pt[0] * np.sin(rad) + center)])
    return np.array(newpts, dtype='float64')


def compute_nme(preds, target):
    interocular = np.linalg.norm(target[0,] - target[10,])
    nme = np.sum(np.linalg.norm(preds - target, axis=1)) / (interocular * 11)
    return nme
