import os

DataBaseDir = '../data'
ModelRootDir = '%s/model' % DataBaseDir

## configurations for test
nsfw_model_weight_file = '%s/nsfw/weight/open_nsfw-weights.npy' % ModelRootDir

TestOutputDir = '../data/test'
if(os.path.exists(TestOutputDir) == False):
    os.makedirs(TestOutputDir)

#strategy = 'attention_nsfw'
strategy = 'zz_nsfw'
batch_size = 16 #128
#num_class = 1
num_class = 3
kfold = 5
kfold_seed = 2018
epochs = 100
learning_rate = 0.0002
debug = True 
sampling_ratio = 0.2

#input_shape = [128, 128, 3]

level_label_dict = {
    'normal': 0,
    'sexual': 1,
    'toxic': 2,
    'unknown': 3,
}

level_en_zn = {
    'normal': '正常',
    'sexual': '性感',
    'toxic': '色情',
    'unknown': 'unknown',
}

level_zn_en = {
    '正常': 'normal',
    '性感': 'sexual',
    '色情': 'toxic',
    'unknown': 'unknown',
}

thresholds = {
    'toxic': 0.65,
    'sexual': 0.60,
    'normal': 0.0
}

tpr_factor = {
    0.3: 0.4,
    0.5: 0.3,
    0.7: 0.3
}

data_set_route = {
    '0819': '%s/raw/8月19日色情图片评估' % DataBaseDir, # sampled from 8/19/2018
    '0819_new': '%s/raw/8月19日色情图片评估_new' % DataBaseDir, # sampled from 8/19/2018
    'history': '%s/raw/色情图片已标记' % DataBaseDir, # toxic info in history
    '1109': '{}/raw/updated_1109'.format(DataBaseDir),
    'test_0819_part1': '{}/raw/test_0819_part1'.format(DataBaseDir),
}
