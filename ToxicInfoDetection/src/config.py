import os

DataBaseDir = '../data'
ModelRootDir = '%s/model' % DataBaseDir

## configurations for test
nsfw_model_weight_file = '%s/nsfw/weight/open_nsfw-weights.npy' % ModelRootDir

TestOutputDir = '../data/test'
if(os.path.exists(TestOutputDir) == False):
    os.makedirs(TestOutputDir)

strategy = 'zz_nsfw'
batch_size = 32 #128
num_class = 3
kfold = 5
kfold_seed = 2018
epochs = 100

input_shape = [256, 256, 3]

level_label_dict = {
    'normal': 0,
    'sexual': 1,
    'toxic': 2,
}

level_en_zn = {
    'normal': '正常',
    'sexual': '性感',
    'toxic': '色情',
}

level_zn_en = {
    '正常': 'normal',
    '性感': 'sexual',
    '色情': 'toxic',
}

data_set_route = {
    '0819': '%s/raw/8月19日色情图片评估' % DataBaseDir, # sampled from 8/19/2018
    '0819_new': '%s/raw/8月19日色情图片评估_new' % DataBaseDir, # sampled from 8/19/2018
    'history': '%s/raw/色情图片已标记' % DataBaseDir, # toxic info in history
}
