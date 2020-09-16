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
import jizhu.config as config


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


def get_info(dicomPath, jsonPath):
    path = "./train.npy" if "150" in dicomPath else "./val.npy"
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


        dcm_paths = glob.glob(os.path.join(dicomPath, "**", "**.dcm"))  # 具体的图片路径
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

        # dirty_data_path = ['E:\BME\competition\spark\data\lumbar_train150\study82\image39.dcm',
        #                    'E:\BME\competition\spark\data\lumbar_train150\study150\image17.dcm']


        # result.drop(result[result['dcmpath'] in dirty_data_path].index)
        # result = result.set_index('dcmPath')['annotation']  # 然后把index设置为路径，值设置为annotation
        result = result[["dcmPath", "annotation"]].values
        np.save(path, result)
    else:
        result = np.load(path, allow_pickle=True)
    return result


def ListSecondMinIndex(lt):
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



def PointToPointDistance(point1,point2):

    # 计算两点间距离
    # point1 = np.array([x1,y1,z1])
    # point2 = np.array([x2,y2,z2])

    return np.square(point1 - point2).sum()



def PointToSurfaceDistance(normal_vector,point_in_surface,point_outside):
    # normal_vector 法向量
    # point_in_surface 面上一点
    # point_outside 面外一点


    D = - normal_vector.dot(point_in_surface)
    # print(D)

    # up = normal_vector * point_outside + D
    # square = np.square(normal_vector)
    # sum = square.sum()

    distance = np.abs(normal_vector.dot(point_outside) + D)/np.sqrt(np.square(normal_vector).sum())

    return distance


def IsAxial(ImageOrientation):

    axial_normal_vector = np.array([0, 0, 1])

    satt_normal_vector = np.array([1, 0, 0])

    ImageOrientation = ImageOrientation.split('\\')
    # print(ImageOrientation)
    # print(type(ImageOrientation[0]))

    slice_vector1 = np.array(ImageOrientation[:3], np.float)
    slice_vector2 = np.array(ImageOrientation[3:], np.float)

    slice_normal_vector = np.cross(slice_vector1, slice_vector2)

    if np.abs(axial_normal_vector.dot(slice_normal_vector)) > 0.7:
        return True
    else:
        return False


def CaculateDisc3DCoordinate(ImagePosition,Disc_landmark,spacing,vector_x,vector_y):
    # ImagePosition:矢状面图在3维坐标系原点坐标[x,y,z]
    # Disc_landmark: 矢状面的标记点坐标:[x,y]
    # spacing:每个像素的真实距离大小[spacing_x,spacing_y]
    # vector_x:矢状面图像的x轴方向向量[x,y,z]
    # vector_y:矢状面图像y轴方向向量[x,y,z]


    Disc_landmark_3DCoordinate = ImagePosition + vector_x * (spacing[0] * Disc_landmark[0]) + vector_y * (spacing[1] * Disc_landmark[1])



    return Disc_landmark_3DCoordinate

def ExtractSattInfo(dicomPath,jsonPath,is_train):

    print(dicomPath)
    if is_train:
        path = "train"
    else:
        path = "test"

    annotation_info = pd.DataFrame(columns=('studyUid', 'seriesUid', 'instanceUid', 'annotation'))

    print(jsonPath)
    json_df = pd.read_json(jsonPath)

    # print(json_df)

    for idx in json_df.index:
        studyUid = json_df.loc[idx, "studyUid"]
        seriesUid = json_df.loc[idx, "data"][0]['seriesUid']
        instanceUid = json_df.loc[idx, "data"][0]['instanceUid']
        annotation = json_df.loc[idx, "data"][0]['annotation']
        row = pd.Series(
            {'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid, 'annotation': annotation})
        annotation_info = annotation_info.append(row, ignore_index=True)

    dcm_paths = glob.glob(os.path.join(dicomPath, "**", "**.dcm"))  # 具体的图片路径
    # print(dcm_paths)
    # 'studyUid','seriesUid','instanceUid'

    # 定位图：0020|0032 Image Position;0020|0037 Image Orientation Patient;0028|0030 pixel spacing;
    # 切片图：以上 +  0028|0010 rows;0028|0011 columns

    # ["检查实例号：唯一标记不同检查的号码.", "序列实例号：唯一标记不同序列的号码.", "SOP实例"]

    tag_list = ['0020|000d', '0020|000e', '0008|0018','0020|0032','0020|0037','0028|0030','0028|0010','0028|0011']
    dcm_info = pd.DataFrame(columns=('dcmPath', 'studyUid', 'seriesUid', 'instanceUid'))
    for dcm_path in dcm_paths:

        try:
            studyUid, seriesUid, instanceUid,ImagePosition,ImageOrientation,pixelspacing,Rows,Columns = dicom_metainfo(dcm_path, tag_list)
            # CaculateDisc3DCoordinate(ImagePosition)

            row = pd.Series(
                {'dcmPath': dcm_path, 'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid,
                 'ImagePosition':ImagePosition,'ImageOrientation':ImageOrientation,'pixelspacing':pixelspacing,
                 'Rows':Rows,'Columns':Columns})

            print("try: ", dcm_path)
            dcm_info = dcm_info.append(row, ignore_index=True)
        except:
            # print("except: ", dcm_path)
            continue

        result = pd.merge(annotation_info, dcm_info, on=['studyUid', 'seriesUid', 'instanceUid'])

        # result = result.set_index('dcmPath')['annotation']  # 然后把index设置为路径，值设置为annotation


        np.save(os.path.join(config.external_data_path,'./result_%s.npy'%path), result.to_dict(orient = 'records') )
        result.to_csv(os.path.join(config.external_data_path,'result_%s.csv'%path))

        # result = result[["dcmPath", "annotation"]].values

    # return result

def CreateAxialCsv(dicomPath,is_train = True):

    #取出数据集中所有的Axial data 生成一个CSV

    if is_train:
        path = "train"
    else:
        path = "test"

    dcm_paths = glob.glob(os.path.join(dicomPath, "**", "**.dcm"))  # 具体的图片路径
    # print(dcm_paths)
    # 'studyUid','seriesUid','instanceUid'
    # 定位图：0020|0032 Image Position;0020|0037 Image Orientation Patient;0028|0030 pixel spacing;
    # 切片图：以上 +  0028|0010 rows;0028|0011 columns

    # ["检查实例号：唯一标记不同检查的号码.", "序列实例号：唯一标记不同序列的号码.", "SOP实例"]
    tag_list = ['0020|000d', '0020|000e', '0008|0018', '0020|0032', '0020|0037', '0028|0030', '0028|0010', '0028|0011']
    dcm_info = pd.DataFrame(columns=('dcmPath', 'studyUid', 'seriesUid', 'instanceUid'))
    for dcm_path in dcm_paths:
        try:
            print("try: ", dcm_path,end='')
            studyUid, seriesUid, instanceUid, ImagePosition, ImageOrientation, pixelspacing, Rows, Columns = dicom_metainfo(
                dcm_path, tag_list)

            if IsAxial(ImageOrientation):
                print('Is Axial')
                row = pd.Series(
                    {'dcmPath': dcm_path, 'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid,
                     'ImagePosition': ImagePosition, 'ImageOrientation': ImageOrientation, 'pixelspacing': pixelspacing,
                     'Rows': Rows, 'Columns': Columns,'identification':'','disc_dcmPath':'','label':'','disc_instanceUid':''})
                dcm_info = dcm_info.append(row, ignore_index=True)
            print('')
        except:
            print("except: ", dcm_path)
            continue

    print(path)

    np.save(os.path.join(config.external_data_path,'dcm_info_%s.npy'%path),dcm_info.to_dict(orient= 'records'))
    dcm_info.to_csv(os.path.join(config.external_data_path,'dcm_info_%s.csv'%path))




def CreatAxialDataset(dicomPath,jsonPath,is_train = True):
    # dicomPath:train或者val的根目录
    # jsonPath:标注文�
    path = "train"
    if is_train == True:
        path = "train"
    elif is_train == False:
        path = "test"
    if not os.path.exists(os.path.join(config.external_data_path,"axial_info_%s.npy"%path)):

        ExtractSattInfo(dicomPath ,jsonPath,is_train = is_train)
        CreateAxialCsv(dicomPath,is_train = is_train)

        result = np.load(os.path.join(config.external_data_path,'result_%s.npy'%path), allow_pickle=True)
        axial_result = np.load(os.path.join(config.external_data_path,'dcm_info_%s.npy'%path), allow_pickle=True)

        axial_result_csv = pd.read_csv(os.path.join(config.external_data_path,'dcm_info_%s.csv'%path))

        # print(result)
        pre_axial_data = None

        for study in result:
            # print(study['annotation'][0]['data'])
            # break

            mask = []
            for i,axial in enumerate(axial_result):
                if axial['studyUid'] == study['studyUid']:
                    mask.append(i)
            # mask = axial_result[:]['studyUid'] == study['studyUid']

            #取出该study下所有的轴状图
            study_all_axial = axial_result[mask]


            # print(study_all_axial)

            annotation = study['annotation']

            # print(annotation[0])
            # print(annotation[0]['data'])
            #
            # if is_train:
            points = annotation[0]['data']['point']
            # else:
            #     points = annotation[0]['point']
            #
            # print(points)


            for point in points:
                #逐个点进行遍历

                # print(point)
                # if 'disc' in point['tag'].keys():

                    # print(point)

                ImagePosition = np.array(study['ImagePosition'].split('\\'),np.float)

                Disc_landmark = [point['coord'][0],point['coord'][1]]

                spacing = np.array(study['pixelspacing'].split('\\'),np.float)

                direct_vector = np.array(study['ImageOrientation'].split('\\'),np.float)

                vector_x , vector_y = direct_vector[:3],direct_vector[3:]

                Disc_landmark_3DCoordinate = CaculateDisc3DCoordinate(ImagePosition = ImagePosition,
                                         Disc_landmark = Disc_landmark,
                                         spacing = spacing,
                                         vector_x = vector_x,
                                         vector_y = vector_y)

                disc_to_all_axial_distance = []
                print(len(study_all_axial))
                for i,axial in enumerate(study_all_axial):
                    #对该disc点计算本study下所有的axial图像距离

                    # print("",i)
                    print("index: %d,axialPath: %s "%(i,axial['dcmPath']),end = '')

                    ImagePosition = np.array(axial['ImagePosition'].split('\\'), np.float)


                    spacing = np.array(axial['pixelspacing'].split('\\'), np.float)

                    direct_vector = np.array(axial['ImageOrientation'].split('\\'), np.float)

                    vector_x, vector_y = direct_vector[:3], direct_vector[3:]

                    normal_vector = np.cross(vector_x,vector_y)

                    distance = PointToSurfaceDistance(normal_vector = normal_vector,
                                                      point_in_surface = ImagePosition,
                                                      point_outside = Disc_landmark_3DCoordinate)

                    # print("distance :",distance)
                    disc_to_all_axial_distance.append(distance)


                print(disc_to_all_axial_distance)

                disc_to_all_axial_distance = np.array(disc_to_all_axial_distance)
                #找到距离最小的轴状图，记录其路径
                disc_to_all_axial_distance_index = np.argsort(disc_to_all_axial_distance)

                min_distant_index = disc_to_all_axial_distance_index[0]
                second_min_distant_index = disc_to_all_axial_distance_index[1]


                if is_train:

                    # 取多少个数据
                    # for min_index in [min_distant_index,second_min_distant_index]:
                    for min_index in [min_distant_index]:
                        axial_path  = study_all_axial[min_index]['dcmPath']

                        # print("find min: ",min(disc_to_all_axial_distance))
                        # print("axial_path: ",axial_path)
                        # print("satt_path: ",study['dcmPath'])
                        # print(axial_result_csv.disc_dcmPath[axial_result_csv['dcmPath'] == axial_path])

                        # break
                        axial_result_csv.loc[axial_result_csv['dcmPath'] == axial_path,'disc_dcmPath']   = study['dcmPath']
                        axial_result_csv.loc[axial_result_csv['dcmPath'] == axial_path, 'disc_instanceUid'] = study['instanceUid']
                        axial_result_csv.loc[axial_result_csv['dcmPath'] == axial_path,'identification'] = point['tag']['identification']
                        if 'disc' in point['tag'].keys():
                            if point['tag']['disc'] == '':
                                axial_result_csv.loc[axial_result_csv['dcmPath'] == axial_path, 'label'] = 'v1'
                            else:
                                axial_result_csv.loc[axial_result_csv['dcmPath'] == axial_path,'label']          = point['tag']['disc']
                        else:
                            if point['tag']['vertebra'] == '':
                                axial_result_csv.loc[axial_result_csv['dcmPath'] == axial_path, 'label'] = 'v1'
                            else:
                                axial_result_csv.loc[axial_result_csv['dcmPath'] == axial_path, 'label'] = point['tag']['vertebra']

                else:
                    for index in disc_to_all_axial_distance_index:

                        axial_path = study_all_axial[index]['dcmPath']

                        print("is null: ",
                              axial_result_csv.loc[axial_result_csv['dcmPath'] == axial_path, 'identification'].iloc[0])

                        if pd.isnull(axial_result_csv.loc[axial_result_csv['dcmPath'] == axial_path, 'identification'].iloc[0]):

                            break

                    # print("find min: ",min(disc_to_all_axial_distance))
                    # print("axial_path: ",axial_path)
                    # print("satt_path: ",study['dcmPath'])
                    # print(axial_result_csv.disc_dcmPath[axial_result_csv['dcmPath'] == axial_path])

                    # break
                    axial_result_csv.loc[axial_result_csv['dcmPath'] == axial_path, 'disc_dcmPath'] = study['dcmPath']
                    axial_result_csv.loc[axial_result_csv['dcmPath'] == axial_path, 'disc_instanceUid'] = study['instanceUid']
                    axial_result_csv.loc[axial_result_csv['dcmPath'] == axial_path, 'identification'] = point['tag']['identification']




            # print(axial_result_csv)

        # axial_result_csv.drop(columns=['Unnamed: 0'], inplace=True)

        # axial_result_csv['label'].fillna(value='v1', inplace=True)
        if is_train:
            axial_result_csv.dropna(axis=0, how='any', subset=['disc_dcmPath', 'identification', 'label'], inplace=True)
        else:
            axial_result_csv.dropna(axis=0, how='any', subset=['disc_dcmPath'], inplace=True)
            # pass
        # axial_result_csv.dropna(axis=0, how='any', inplace=True)

        axial_result_csv.reset_index(drop=True, inplace=True)


        # print(axial_result_csv)


        print(path)

        axial_result_csv.to_csv(os.path.join(config.external_data_path,'axial_info_%s.csv'%path))

        axial_result_dict = axial_result_csv.iloc[:, 1:].to_dict(orient = 'records')

        np.save(os.path.join(config.external_data_path,'axial_info_%s.npy'%path), axial_result_dict)

    else:
        axial_result_dict = np.load(os.path.join(config.external_data_path,'axial_info_%s.npy'%path),allow_pickle = True)
        axial_result_csv = pd.read_csv(os.path.join(config.external_data_path,'axial_info_%s.csv'%path))


    return axial_result_dict , axial_result_csv


def AddcsvInfo(bigdataframe,data_row,datapath,datalist =  ['disc_dcmPath','identification','label'] ):


    for data in datalist:
        bigdataframe.loc[bigdataframe['dcmPath'] == datapath,data ] = data_row[data]

    return bigdataframe



def AddExtraAxialData(dcm_info_csv_path,axial_info_csv_path):


    dcm_info = pd.read_csv(dcm_info_csv_path)
    axial_info = pd.read_csv(axial_info_csv_path)

    # print("dcm_info",dcm_info)
    # mask = axial_info['Unnamed: 0']
    # dcm_info.iloc[mask,:] = axial_info.drop(columns = ['Unnamed: 0'])

    for index ,row in axial_info.iterrows():
        dcm_path = row['dcmPath']
        dcm_info.loc[dcm_info['dcmPath'] == dcm_path,'disc_dcmPath'] = row['disc_dcmPath']
        dcm_info.loc[dcm_info['dcmPath'] == dcm_path, 'identification'] = row['identification']
        dcm_info.loc[dcm_info['dcmPath'] == dcm_path, 'label'] = row['label']


    # print(dcm_info)



    pre_studyid = None
    for index,data in dcm_info.iterrows():

        #按studyid 进行遍历
        # print("data: ",data)
        studyid = data['studyUid']

        if studyid == pre_studyid:
            continue
        pre_studyid = studyid

        dcm_study_part = dcm_info[dcm_info['studyUid'] == studyid]

        dcm_study_part.reset_index(drop = False,inplace = True)
        print("studyID: ",studyid)
        # print(dcm_study_part)
        # axial_study_part = axial_info[axial_info['studyUid'] == studyid]
        count = 0
        while dcm_study_part['label'].isnull().any():
            count = count + 1
            if count > 10:
                count = 0
                break
            # print(dcm_study_part)
            #
            # print("*************************************************************************")

            for j,part_data in dcm_study_part.iterrows():
                #study_part的每一个项进行遍历
                if j==0 or j==len(dcm_study_part)-1 or pd.isna(part_data['label']):
                    continue

                print("******************************************")
                print("dcm study part")
                print(dcm_study_part)

                print("part_data")
                print(part_data)

                print("j:",j)
                print("len dcmstudypart: ",len(dcm_study_part))

                print("*****************************************")
                pre_data = dcm_study_part.iloc[j-1,:]
                next_data = dcm_study_part.iloc[j+1,:]

                # print(pre_data)
                # print(part_data['ImagePosition'].split('\\'))

                point_pre = np.array(pre_data['ImagePosition'].split('\\'),np.float)
                point_now = np.array(part_data['ImagePosition'].split('\\'),np.float)
                point_next = np.array(next_data['ImagePosition'].split('\\'),np.float)

                d1 = PointToPointDistance(point_pre,point_now)
                d2 = PointToPointDistance(point_now,point_next)


                if  pd.isna(pre_data['label']) and  pd.isna(next_data['label']):
                    if d1<d2:
                        pre_data['label'] = part_data['label']
                        datapath = pre_data['dcmPath']

                        dcm_study_part = AddcsvInfo(dcm_study_part,part_data,datapath)


                    else:
                        # print(next_data['label'])
                        next_data['label'] = part_data['label']
                        datapath = next_data['dcmPath']
                        dcm_study_part = AddcsvInfo(dcm_study_part, part_data, datapath)
                        print(dcm_study_part.loc[dcm_study_part['dcmPath'] == datapath,'label'])

                elif not pd.isna(pre_data['label']) and not pd.isna(next_data['label']):
                    continue

                elif not pd.isna(pre_data['label']) and pre_data['label'] != part_data['label']:
                    next_data['label'] = part_data['label']
                    datapath = next_data['dcmPath']
                    dcm_study_part = AddcsvInfo(dcm_study_part, part_data, datapath)


                elif pre_data['label'] == part_data['label']:
                    if np.abs(d1 - d2) < 2:
                        next_data['label'] = part_data['label']
                        datapath = next_data['dcmPath']
                        dcm_study_part = AddcsvInfo(dcm_study_part, part_data, datapath)

                elif not pd.isna(next_data['label']) and next_data['label'] != part_data['label']:
                    pre_data['label'] = part_data['label']
                    datapath = pre_data['dcmPath']

                    dcm_study_part = AddcsvInfo(dcm_study_part, part_data, datapath)


                elif next_data['label'] == part_data['label']:
                    if np.abs(d1 - d2) < 2:
                        pre_data['label'] = part_data['label']
                        datapath = pre_data['dcmPath']

                dcm_study_part = AddcsvInfo(dcm_study_part,part_data,datapath)
                # print("************************************")
                #
                # print("study_part")
                # print(dcm_study_part)

                for index, row in dcm_study_part.iterrows():
                    dcm_path = row['dcmPath']
                    dcm_info.loc[dcm_info['dcmPath'] == dcm_path, 'disc_dcmPath'] = row['disc_dcmPath']
                    dcm_info.loc[dcm_info['dcmPath'] == dcm_path, 'identification'] = row['identification']
                    dcm_info.loc[dcm_info['dcmPath'] == dcm_path, 'label'] = row['label']

                # print("dcm_info")
                # print(dcm_info)
                #
                # print("**************************************")

    dcm_info.to_csv("new_axial.csv")
            # print(dcm_study_part)



def CreatPointToAxialCsv(result_dict_path,all_axial_csv_path,is_train):

    result = np.load(result_dict_path,allow_pickle= True)
    all_axial_csv = pd.read_csv(all_axial_csv_path)

    shape_list = []
    # print(type(img))
    # print(img.shape)

    # print(self.all_data_dict)
    # 获取某个矢状面的数据
    point_to_axial_dataframe = pd.DataFrame(columns=('studyUid', 'seriesUid', 'instanceUid',
                                                     'coord','dcmPath','identification',
                                                     'disc','vertebra','label',
                                                     'axial_path'))

    # dcm_info = pd.DataFrame(columns=('dcmPath', 'studyUid', 'seriesUid', 'instanceUid'))

    # point_to_axial_dataframe = pd.DataFrame()

    for study in result:
        # print(study['annotation'][0]['data'])
        # break
        studyUid = study['studyUid']
        seriesUid = study['seriesUid']
        instanceUid = study['instanceUid']

        # mask = []
        # for i, axial in enumerate(all_axial_csv):
        #     if axial['studyUid'] == study['studyUid']:
        #         mask.append(i)
        # mask = axial_result[:]['studyUid'] == study['studyUid']


        # 取出该study下所有的轴状图
        study_all_axial = all_axial_csv.loc[all_axial_csv['studyUid'] == studyUid,:]

        # print(study_all_axial)

        annotation = study['annotation']

        # print(annotation[0])
        # print(annotation[0]['data'])
        #
        # if is_train:
        points = annotation[0]['data']['point']
        # else:
        #     points = annotation[0]['point']
        #
        # print(points)


        for point in points:
            disc = 0
            vertebra = 0
            label = None
            # 逐个点进行遍历

            # print(point)
            # if 'disc' in point['tag'].keys():

            # print(point)

            if 'disc' in point['tag'].keys():
                disc = 1
                vertebra = 0
                if point['tag']['disc'] == '':
                    label = 'v1'
                else:
                    label = point['tag']['disc']
            elif 'vertebra' in point['tag'].keys():
                disc = 0
                vertebra = 1

                if point['tag']['vertebra'] == '':
                    label = 'v1_v'
                else:
                    if is_train:
                        label = point['tag']['vertebra']
                    else:
                        label = point['tag']['vertebra']


            coord = point['coord']
            identification = point['tag']['identification']
            dcmPath = study['dcmPath']

            ImagePosition = np.array(study['ImagePosition'].split('\\'), np.float)

            Disc_landmark = [point['coord'][0], point['coord'][1]]

            spacing = np.array(study['pixelspacing'].split('\\'), np.float)

            direct_vector = np.array(study['ImageOrientation'].split('\\'), np.float)

            vector_x, vector_y = direct_vector[:3], direct_vector[3:]


            Disc_landmark_3DCoordinate = CaculateDisc3DCoordinate(ImagePosition=ImagePosition,
                                                                  Disc_landmark=Disc_landmark,
                                                                  spacing=spacing,
                                                                  vector_x=vector_x,
                                                                  vector_y=vector_y)




            disc_to_all_axial_distance = []


            for i, axial in study_all_axial.iterrows():
                # 对该disc点计算本study下所有的axial图像距离

                # print("",i)
                # print("index: %d,axialPath: %s " % (i, axial['dcmPath']))

                ImagePosition = np.array(axial['ImagePosition'].split('\\'), np.float)

                spacing = np.array(axial['pixelspacing'].split('\\'), np.float)

                direct_vector = np.array(axial['ImageOrientation'].split('\\'), np.float)

                vector_x, vector_y = direct_vector[:3], direct_vector[3:]

                normal_vector = np.cross(vector_x, vector_y)

                distance = PointToSurfaceDistance(normal_vector=normal_vector,
                                                  point_in_surface=ImagePosition,
                                                  point_outside=Disc_landmark_3DCoordinate)

                # print("distance :",distance)
                disc_to_all_axial_distance.append(distance)

            # print(disc_to_all_axial_distance)

            disc_to_all_axial_distance = np.array(disc_to_all_axial_distance)
            # 找到距离最小的轴状图，记录其路径
            disc_to_all_axial_distance_index = np.argsort(disc_to_all_axial_distance)

            min_distant_index = disc_to_all_axial_distance_index[0]
            second_min_distant_index = disc_to_all_axial_distance_index[1]


            axial_path = study_all_axial.iloc[min_distant_index]['dcmPath']

            row = pd.Series({'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid,
                 'coord':coord,'identification':identification,'disc':disc,
                 'vertebra':vertebra,'label':label,'dcmPath':dcmPath,
                 'axial_path':axial_path})

            print(row)
            point_to_axial_dataframe = point_to_axial_dataframe.append(row,ignore_index=True)


            # print(point_to_axial_dataframe)



    point_to_axial_dataframe.to_csv(os.path.join(config.external_data_path,'point_to_axial.csv'))
    point_to_axial_dict = point_to_axial_dataframe.to_dict(orient='records')
    np.save(os.path.join(config.external_data_path,'point_to_axial_dict.npy'), point_to_axial_dict)


    return point_to_axial_dataframe,point_to_axial_dict


if __name__ == '__main__':
    # dcm_path = r'E:/DATA/Spart_AI/lumbar_train51/train/study0/*.dcm'
    # file_list = glob.glob(dcm_path)

    pd.set_option('expand_frame_repr', False)


    info_dict = {
        "vertebra": {"v1": "正常", "v2": "退行性改变"},
        "disc": {"v1": "正常", "v2": "膨出", "v3": "突出", "v4": "脱出", "v5": "椎体内疝出"}
    }

    trainPath = config.trainPath
    trainjsonPath = config.trainjsonPath

    valPath = config.valPath
    valjsonPath = config.valjsonPath

    testPath = config.testPath
    testjsonPath = config.testjsonPath


    test_result_dict_Path = r'D:\project\zjx\competitions\spark\code\utils\result_test.npy'
    test_all_axial_Path = r'D:\project\zjx\competitions\spark\code\utils\dcm_info_test.csv'


    # ####################################  #用于生成测试用轴状图数据 Csv 和 npy############################################################
    CreatAxialDataset(dicomPath=testPath, jsonPath=testjsonPath, is_train=False)


    csv,dict = CreatPointToAxialCsv(result_dict_path = test_result_dict_Path,
                                    all_axial_csv_path = test_all_axial_Path,
                                    is_train = False)



    # ####################################  #用于生成轴状图数据 Csv 和 npy############################################################

    #
    CreatAxialDataset(dicomPath = config.trainPath,jsonPath = config.trainjsonPath , is_train = True)
    CreatAxialDataset(dicomPath= config.valPath, jsonPath=config.valjsonPath, is_train=True)
    # # CreatAxialDataset(dicomPath=testPath, jsonPath=testjsonPath, is_train=False)

    # ##用于可视化关键点####
    # 可视化部分

    # ####################################  #用于生成轴状图数据 Csv 和 npy############################################################

    # for path in [[trainPath,trainjsonPath],[valPath,valjsonPath]]:
    #
    #     result = get_info(path[0], path[1])
    #
    #     pd.set_option('expand_frame_repr', False)
    #
    #     # distance = PointToSurfaceDistance(normal_vector,point_in_surface,point_outside)
    #
    #     # 定位图：0020|0032 Image Position;    0020|0037 Image Orientation Patient;    0028|0030 pixel spacing;
    #     # 切片图：以上 +  0028|0010 rows;0028|0011 columns
    #
    #     axial_result = CreatAxialDataset(dicomPath=path[0], jsonPath=path[1])

    # #####################################用于生成轴状图数据 Csv 和 npy############################################################

    # ####################################            合并数据        ############################################################

    pd.set_option('expand_frame_repr', False)

    # train_json = pd.read_json(config.trainjsonPath)



    train_csv = pd.read_csv(r'D:\project\zjx\competitions\spark\code\utils\axial_info_train.csv')
    val_csv = pd.read_csv(r'D:\project\zjx\competitions\spark\code\utils\axial_info_val.csv')

    # result_test = np.load('result_test.npy')
    # result_train = np.load('result_train.npy')

    frames = [train_csv, val_csv]

    all_csv = pd.concat(frames)

    all_csv.reset_index(drop=True, inplace=True)
    all_csv.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], inplace=True)

    all_csv.to_csv('train_all.csv')
    all_dict = all_csv.to_dict(orient='records')
    np.save('train_all.npy',all_dict)


    ####################################            合并数据        ############################################################




    # # result[:][:] = result
    # # result = np.squeeze(result)
    # # print(result[0])
    # print(result[0][1][0]['data']['point'][0]['tag'])
    # print(result[0][0])
    # print(len(result))
    # print(type(result))
    # # print(type(result))
    # # print(result)
    #
    #
    # print(len(result))
    # for i in range(len(result)):
    #     img_dir = result[i][0]  # 获取图片的地址
    #     print(img_dir)
    #     img_arr = dicom2array(img_dir)  # 获取具体的图片数据，二维数据
    #     if len(img_arr.shape) == 2:
    #         img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
    #     annotation = result[i][1]  # 获取图片的标签
    #     tags = annotation[0]['data']['point']
    #     for tag in tags:
    #         coord = tag['coord']
    #         center_x, center_y = coord[0], coord[1]
    #
    #         identification = tag["tag"]['identification']
    #         tag = tag["tag"]
    #         if "vertebra" in tag.keys():
    #             # 椎体
    #             cv2.circle(img_arr, (center_x, center_y), 8, (127, 127, 255))
    #             vertebra = tag['vertebra']
    #             text = f"{vertebra} {'Null' if vertebra=='' else info_dict['vertebra'][vertebra]}|{identification}"
    #             img_arr = cv2ImgAddText(img_arr, text, center_x + 20, center_y - 10, textColor=(255, 0, 0))
    #         else:
    #             # 椎间盘
    #             cv2.circle(img_arr, (center_x, center_y), 8, (127, 255, 127))
    #             disc = tag['disc']
    #             if "," in disc:
    #                 print(disc)
    #                 text = f"{disc} {','.join([info_dict['disc'][d] for d in disc.split(',')])}|{identification}"
    #                 delay = 2
    #             else:
    #                 text = f"{disc} {info_dict['disc'][disc]}|{identification}"
    #             img_arr = cv2ImgAddText(img_arr, text, center_x + 20, center_y - 10, textColor=(0, 255, 0))
    #     cv2.imshow("", cv2.resize(img_arr, (512, 512)))
    #     cv2.waitKey(0)
    #
    #     ##用于可视化关键点####





