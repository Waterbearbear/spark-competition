


#train parameters
trainPath = r'E:\BME\competition\spark\data\lumbar_train150'
trainjsonPath = r'E:\BME\competition\spark\data\lumbar_train150_annotation.json'


valPath = r'E:\BME\competition\spark\data\lumbar_train51'
valjsonPath = r'E:\BME\competition\spark\data\lumbar_train51_annotation.json'

isTrain = True
epochs = 10
eval_epochs = 1

batch_size =  8
lr = 5e-3

n_threads = 1
gpu = True

#adam momentum
beta1 = 0.5

#heatmap parameters
sigma = 1.25

#img and label
num_classes = 5
output_size = [64,64]
input_size  = [256,256]
num_landmark = 11

checkpoints_dir = r"E:\BME\competition\spark\code\checkpoint"
