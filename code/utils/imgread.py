# -*- coding: utf-8 -*- 
# @Time 2020/6/15 9:53
# @Author wcy
import glob
import os
from tqdm import tqdm
import SimpleITK as sitk
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 127), textSize=14):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def dicom_metainfo(dicm_path, list_tag):
    '''
    获取dicom的元数据信息
    :param dicm_path: dicom文件地址
    :param list_tag: 标记名称列表,比如['0008|0018',]
    :return:
    '''
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    reader.SetFileName(dicm_path)
    reader.ReadImageInformation()
    return [reader.GetMetaData(t) for t in list_tag]


def dicom2array(dcm_path):
    '''
    读取dicom文件并把其转化为灰度图(np.array)
    https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html
    :param dcm_path: dicom文件
    :return:
    '''
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetImageIO('GDCMImageIO')
    image_file_reader.SetFileName(dcm_path)
    image_file_reader.ReadImageInformation()
    image = image_file_reader.Execute()
    if image.GetNumberOfComponentsPerPixel() == 1:
        image = sitk.RescaleIntensity(image, 0, 255)
        if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
            image = sitk.InvertIntensity(image, maximum=255)
        image = sitk.Cast(image, sitk.sitkUInt8)
    img_x = sitk.GetArrayFromImage(image)[0]
    return img_x


def get_info(trainPath, jsonPath):
    path = "./train.npy" if "150" in trainPath else "./vali.npy"
    if not os.path.exists(path):
        annotation_info = pd.DataFrame(columns=('studyUid', 'seriesUid', 'instanceUid', 'annotation'))
        json_df = pd.read_json(jsonPath)
        for idx in tqdm(json_df.index):
            studyUid = json_df.loc[idx, "studyUid"]
            seriesUid = json_df.loc[idx, "data"][0]['seriesUid']
            instanceUid = json_df.loc[idx, "data"][0]['instanceUid']
            annotation = json_df.loc[idx, "data"][0]['annotation']
            row = pd.Series(
                {'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid, 'annotation': annotation})
            annotation_info = annotation_info.append(row, ignore_index=True)
        dcm_paths = glob.glob(os.path.join(trainPath, "**", "**.dcm"))  # 具体的图片路径
        print(dcm_paths)
        # 'studyUid','seriesUid','instanceUid'

        #定位图：0020|0032 Image Position;0020|0037 Image Orientation Patient;0028|0030 pixel spacing;
        #切片图：以上 +  0028|0010 rows;0028|0011 columns

        # ["检查实例号：唯一标记不同检查的号码.", "序列实例号：唯一标记不同序列的号码.", "SOP实例"]

        tag_list = ['0020|000d', '0020|000e', '0008|0018']
        dcm_info = pd.DataFrame(columns=('dcmPath', 'studyUid', 'seriesUid', 'instanceUid'))
        for dcm_path in tqdm(dcm_paths):
            try:
                studyUid, seriesUid, instanceUid = dicom_metainfo(dcm_path, tag_list)
                row = pd.Series(
                    {'dcmPath': dcm_path, 'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid})

                print("try: ",dcm_path)
                dcm_info = dcm_info.append(row, ignore_index=True)
            except:
                print("except: ",dcm_path)
                continue

        result = pd.merge(annotation_info, dcm_info, on=['studyUid', 'seriesUid', 'instanceUid'])
        # result = result.set_index('dcmPath')['annotation']  # 然后把index设置为路径，值设置为annotation
        result = result[["dcmPath", "annotation"]].values
        np.save(path, result)
    else:
        result = np.load(path, allow_pickle=True)
    return result


if __name__ == '__main__':
    # dcm_path = r'E:/DATA/Spart_AI/lumbar_train51/train/study0/*.dcm'
    # file_list = glob.glob(dcm_path)
    info_dict = {
        "vertebra": {"v1": "正常", "v2": "退行性改变"},
        "disc": {"v1": "正常", "v2": "膨出", "v3": "突出", "v4": "脱出", "v5": "椎体内疝出"}
    }
    # 可视化部分
    valiPath = r'E:\BME\competition\spark\data\train'
    valijsonPath = r'E:\BME\competition\spark\data\lumbar_train51_annotation.json'

    trainPath = r'E:\BME\competition\spark\data\lumbar_train150'
    trainjsonPath = r'E:\BME\competition\spark\data\lumbar_train150_annotation.json'

    # result = get_info(trainPath, trainjsonPath)  #获取图片路径及对应的annotation
    result = get_info(trainPath, trainjsonPath)
    # result[:][:] = result
    # result = np.squeeze(result)
    # print(result[0])
    print(result[0][1][0]['data']['point'][0]['tag'])
    print(result[0][0])
    print(len(result))
    print(type(result))
    # print(type(result))
    # print(result)


    print(len(result))
    for i in range(len(result)):
        img_dir = result[i][0]  # 获取图片的地址
        print(img_dir)
        img_arr = dicom2array(img_dir)  # 获取具体的图片数据，二维数据
        if len(img_arr.shape) == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
        annotation = result[i][1]  # 获取图片的标签
        tags = annotation[0]['data']['point']
        for tag in tags:
            coord = tag['coord']
            center_x, center_y = coord[0], coord[1]

            identification = tag["tag"]['identification']
            tag = tag["tag"]
            if "vertebra" in tag.keys():
                # 椎体
                cv2.circle(img_arr, (center_x, center_y), 8, (127, 127, 255))
                vertebra = tag['vertebra']
                text = f"{vertebra} {'Null' if vertebra=='' else info_dict['vertebra'][vertebra]}|{identification}"
                img_arr = cv2ImgAddText(img_arr, text, center_x + 20, center_y - 10, textColor=(255, 0, 0))
            else:
                # 椎间盘
                cv2.circle(img_arr, (center_x, center_y), 8, (127, 255, 127))
                disc = tag['disc']
                if "," in disc:
                    print(disc)
                    text = f"{disc} {','.join([info_dict['disc'][d] for d in disc.split(',')])}|{identification}"
                    delay = 2
                else:
                    text = f"{disc} {info_dict['disc'][disc]}|{identification}"
                img_arr = cv2ImgAddText(img_arr, text, center_x + 20, center_y - 10, textColor=(0, 255, 0))
        cv2.imshow("", cv2.resize(img_arr, (512, 512)))
        cv2.waitKey(0)
