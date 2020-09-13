#train parameters
trainPath = r"/tcdata/data/round2train/train"
trainjsonPath = r'/tcdata/data/round2train_checked.json'


# valPath = r'/tcdata/Data/DatasetA/train/lumbar_train51'
# valjsonPath = r'/tcdata/Data/DatasetA/train/lumbar_train51_annotation.json'


train_allPath = None
train_alljsonPath = None

testPath = r'/tcdata/data/round2test'

testjsonPath = r'/code_zjx_round2_pytorch/External/test_new.json'

testMapjsonPath = r'/tcdata/data/testB50_series_map.json'

test_Dict_Path = r'/code_zjx_round2_pytorch/External/point_to_axial_dict.npy'
test_csv_Path = r'/code_zjx_round2_pytorch/External/point_to_axial.csv'


external_data_path = r'/code_zjx_round2_pytorch/External'

SeriseDescription_dirtycase = ['SCOUT ','3-Plane Loc ','01 Scano GE ST','02 Scano SE CS','3-pl Loc GRE',
                               'FGR ','Scoutview','SHIM2D','ScoutA_TSC 100/18 20mm','SURVEY','localizer ',
                               '03 Scano S/T/C','02 Scano COR/TRS']

all_SeriseDescription = {'Sag FSE2D T1W ', 'T2WI_TRA', 'T2WI-TRAc ', 'STIR_TRA', '3-pl Loc GRE', 'IRFSE 4000/90/123 5mmS', '??T1_FSE',
 'T2WI_TRAc ', 'FL08/FL08/OAX FRFSE T2', 'Sag STIR2D T2W', 'SG T2W', 'FST2_SAGc ', 'SG T1W', 't2_tse_tra_msma ',
 't2_tse_tra_msma_320 ', 'T2_FSE_5mm(S) ', 'T2WI-TRA', 'FSE 4000/128 5mmS ', '????T2', 'STIR_COR', 'SED ', 'T2WI-SAG',
 'FL08/FL08/OSag FRFSE T2 ', 'AX T2W', 'TIWI/SE/5mm/S/C ', 'T1_SE_5mm(SC) ', 'T2WI_TRAch', 'T1WI-SAG', 'FLAIR_SAG ',
 '01 Scano GE ST', 'OSag STIR ', 'T2WI/FSE/5mmT ', 't2_tse_rst_tra_msma_FIL ', 'SCOUT ', 'ScoutA_TSC 100/18 20mm',
 '??T2', 'STIR_SAGc ', 't2_tse_sag', 'T2WI_SAG', 'FL08/FL06/OSag FIR (fat suppr ', 'SG T2W Stir ', 'SE 420/19 5mmS',
 'FSE 3200/110 5mmT ', '04 T1 SAG FSE ', 't2_tse_dixon_sag_320_F', '10 T2 SAG ', 't2_tirm_sag ', 'FGR ',
 '02 Scano COR/TRS', '3-Plane Loc ', 'OSag T1 FSE ', '????', 'Axi TR-FSE2D T2W', 't2_tirm_sag_p2', 't2_tirm_tse_sag ',
 'T1WI_SAGc ', 'OAx T2 FSE', '02 Scano SE CS', '11 T2 TRS ', '08 T2 T5mm#11 ', 't1_tse_sag', 't2_tse_sag_384',
 'Sys2DCard ', 'STIR-SAG', 'SHIM2D', 'T2_FSE_5mm(T) ', '07 T2 S5mm#7', 'T2WI_SAGc ', 'Sag TR-FSE2D T2W',
 'WFST2_SAG(W/F)', 't1_tse_sag_384', 't2_tse_rst_sag_FIL', 'T1WI-SAGc ', '03 Scano S/T/C', 't1_tse_sag_320',
 'WFST2_SAG(T2) ', 'T2WI/FSE/5mmS ', 'localizer ', 't2_tse_dixon_sag_320_W', 'OSag T2 FRFSE ',
 'FL08/FL08/OSag SE T1       (S ', 'STIR_SAG', 'T1WI_SAG', 't1_tse_sag_FIL', '04 T1 S5mm#7'}


# pretrainedPath = r'D:/project/zjx/competitions/spark/code/checkpoint/ResNeSt_kfold_balance/ResNeSt101_fold2_best.pth'
pretrainedPath = [None,None]

# model_name = 'DenseNet169'
# model_name = 'ResNet152'
# model_name = 'ResNeSt101'
# model_name = 'ResNeSt50'
# model_name = 'ResNet18'
# model_name = 'ResNet50'
# model_name = 'ResNeSt18'

model_name = 'MobileNet_v2'
#双路输入:轴状图+矢状图
# model_type = 'Double'

#单路输入:仅矢状图
model_type = 'Double'

checkpoints_dir = r"./checkpoint/%s_%s_Weights_2"%(model_name,model_type)

#TEST
# isTrain = False
# pretrained = True

#TRAIN
isTrain = True
pretrained = False
continue_train = True


warmup_period = 5
decay_epoch = 30

k_fold = 5
epochs = 2
eval_epochs = 1
step_print = 10

batch_size =  32
lr = 6e-4

n_threads = 0

gpu = False

# with weights
vertebra_weights = [2.3227, 0.6372]
disc_weights = [0.4690, 0.5896, 0.2984, 3.4952, 6.3826]


# no weights
# vertebra_weights = [1,1]
# disc_weights     = [1,1,1,1,1]


#adam momentum
beta1 = 0.5

#heatmap parameters
sigma = 1.25

#img and label

num_classes = [2,5]
output_size = [64,64]
input_size  = [256,256]
num_landmark = 11

#validation parameter

target_names = ['v1','v2','v3','v4','v5']

#test

test_batch_size = 16
submision_output_file_path = r"/result.json"
models_dir = r"./checkpoint/%s_%s_Weights"%(model_name,model_type)