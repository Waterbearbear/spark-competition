import numpy as np
import pandas as pd
import glob
import os
import tqdm


from utils.imgread import dicom_metainfo

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

def ExtractSattInfo(dicomPath,jsonPath):

    path = "train" if "150" in trainPath else "val"

    annotation_info = pd.DataFrame(columns=('studyUid', 'seriesUid', 'instanceUid', 'annotation'))
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


        np.save('./result_%s.npy'%path, result.to_dict(orient = 'records') )
        result.to_csv('result_%s.csv'%path)

        # result = result[["dcmPath", "annotation"]].values

    # return result

def CreateAxialCsv(dicomPath):

    path = "train" if "150" in trainPath else "val"

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
                     'Rows': Rows, 'Columns': Columns,'identification':'','disc_dcmPath':'','label':''})
                dcm_info = dcm_info.append(row, ignore_index=True)
            print('')
        except:
            print("except: ", dcm_path)
            continue

    np.save('dcm_info_%s.npy'%path,dcm_info.to_dict(orient= 'records'))
    dcm_info.to_csv('dcm_info_%s.csv'%path)




def CreatAxialDataset(dicomPath,jsonPath):
    # dicomPath:train或者val的根目录
    # jsonPath:标注文件

    path = "train" if "150" in trainPath else "val"


    ExtractSattInfo(dicomPath ,jsonPath)
    CreateAxialCsv(dicomPath)

    result = np.load('result_%s.npy'%path, allow_pickle=True)
    axial_result = np.load('dcm_info_%s.npy'%path, allow_pickle=True)

    axial_result_csv = pd.read_csv('dcm_info_%s.csv'%path)

    # print(result)
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
        points = annotation[0]['data']['point']
        #
        # print(points)


        for point in points:
            #逐个点进行遍历

            # print(point)
            if 'disc' in point['tag'].keys():

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
                # print(len(study_all_axial))
                for i,axial in enumerate(study_all_axial):
                    #对该disc点计算本study下所有的axial图像距离

                    # print("",i)
                    # print("index: %d,axialPath: %s "%(i,axial['dcmPath']),end = '')

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


                # print(disc_to_all_axial_distance)
                #找到距离最小的轴状图，记录其路径
                min_distant_index = disc_to_all_axial_distance.index(min(disc_to_all_axial_distance))

                axial_path  = study_all_axial[min_distant_index]['dcmPath']

                # print("find min: ",min(disc_to_all_axial_distance))
                # print("axial_path: ",axial_path)
                # print("satt_path: ",study['dcmPath'])
                # print(axial_result_csv.disc_dcmPath[axial_result_csv['dcmPath'] == axial_path])

                # break
                axial_result_csv.loc[axial_result_csv['dcmPath'] == axial_path,'disc_dcmPath']   = study['dcmPath']
                axial_result_csv.loc[axial_result_csv['dcmPath'] == axial_path,'identification'] = point['tag']['identification']
                axial_result_csv.loc[axial_result_csv['dcmPath'] == axial_path,'label']          = point['tag']['disc']
                if point['tag']['disc'] == '':
                    axial_result_csv.loc[axial_result_csv['dcmPath'] == axial_path,'label']    = 'v1'

        # print(axial_result_csv)

    # axial_result_csv.drop(columns=['Unnamed: 0'], inplace=True)

    # axial_result_csv['label'].fillna(value='v1', inplace=True)
    axial_result_csv.dropna(axis=0, how='any', subset=['disc_dcmPath', 'identification', 'label'], inplace=True)


    # axial_result_csv.dropna(axis=0, how='any', inplace=True)

    axial_result_csv.reset_index(drop=True, inplace=True)


    # print(axial_result_csv)



    axial_result_csv.to_csv('axial_info_%s.csv'%path)

    axial_result_dict = axial_result_csv.iloc[:, 1:].to_dict(orient = 'records')

    np.save('axial_info_%s.npy'%path, axial_result_dict)



if __name__ == '__main__':


    # normal_vector = np.array([3,4,5])
    # point_in_surface = np.array([-3,1,1])
    #
    # point_outside = np.array([2,1,0])

    pd.set_option('expand_frame_repr',False)

    # distance = PointToSurfaceDistance(normal_vector,point_in_surface,point_outside)

    # 定位图：0020|0032 Image Position;    0020|0037 Image Orientation Patient;    0028|0030 pixel spacing;
    # 切片图：以上 +  0028|0010 rows;0028|0011 columns

    trainPath = r'E:\BME\competition\spark\data\lumbar_train150'
    trainjsonPath = r'E:\BME\competition\spark\data\lumbar_train150_annotation.json'

    valPath = r'E:\BME\competition\spark\data\lunbar_train51'
    valjsonPath = r'E:\BME\competition\spark\data\lumbar_train51_annotation.json'

    CreatAxialDataset(dicomPath = trainPath,jsonPath=trainjsonPath)
