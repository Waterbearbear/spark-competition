


#train parameters
trainPath = r'..\data\DatasetA\train\lumbar_train150'
trainjsonPath = r'..\data\DatasetA\train\lumbar_train150_annotation.json'


valPath = r'..\data\DatasetA\train\lumbar_train51'
valjsonPath = r'..\data\DatasetA\train\lumbar_train51_annotation.json'


train_allPath = None
train_alljsonPath = None

testPath = r'..\data\DatasetB\test\lumbar_testB50'
testMapjsonPath = r'..\data\DatasetB\test\testB50_series_map.json'
testjsonPath = r'..\data\DatasetB\test\test_B_2.json'

test_Dict_Path = r'..\data\External\point_to_axial_dict.npy'
test_csv_Path = r'..\data\External\point_to_axial.csv'


external_data_path = r'..\data\External'




# pretrainedPath = r'D:\project\zjx\competitions\spark\code\checkpoint\ResNeSt_kfold_balance\ResNeSt101_fold2_best.pth'
pretrainedPath = [None,None]



# model_name = 'DenseNet169'
# model_name = 'ResNet152'
# model_name = 'ResNeSt101'
# model_name = 'ResNeSt50'
model_name = 'ResNet18'
# model_name = 'ResNet50'
# model_name = 'ResNeSt18'


checkpoints_dir = r".\checkpoint\%s_Double_Weights"%(model_name)


#TEST
# isTrain = False
# pretrained = True

#TRAIN
isTrain = True
pretrained = False


warmup_period = 5
decay_epoch = 30

k_fold = 5
axial_epochs = 120
eval_epochs = 1
step_print = 10

batch_size =  32
lr = 6e-4

n_threads = 0

gpu = False
bf16 = True

# with weights
vertebra_weights = [2.3227848101265822, 0.6371527777777778]
disc_weights = [0.46900958466453674, 0.5895582329317269, 0.2983739837398374, 3.4952380952380953, 6.3826086956521735]


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
submision_output_file_path = r"result.json"
models_dir = r".\checkpoint\ResNet18_Double_weights"

