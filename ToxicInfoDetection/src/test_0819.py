# Created by yuanpingzhou at 11/27/18

import numpy as np
from model import InputType
from zz_nsfw import IMAGE_LOADER_YAHOO
from zz_nsfw import extract_nsfw_features
from zz_nsfw import zz_nsfw_network, zz_metric
import config
from keras.optimizers import Adam

## convert image into feature vector 
nsfw_model_dir = '../data/model/nsfw/savedmodel'
test_image_dir = '../data/raw/test_0819_part3_1'
X_test, y_test, test_image_files = extract_nsfw_features(test_image_dir,
                                        InputType.TENSOR.name.lower(),
                                        IMAGE_LOADER_YAHOO,
                                        '%s/%s' % (nsfw_model_dir, 1),
                                        has_supplement= False,
                                        phase= 'test',
                                        return_image_files= True)

print(np.unique(y_test, return_counts= True))

## load models
model_weight_dir = '../data/model/zz_nsfw/weight'
kfold = 5
models = []
for fold in range(kfold):
    network = zz_nsfw_network(print_network=False)
    network.load_weights('{}/{}.weight.{}'.format(model_weight_dir, 'zz_nsfw', fold))
    opti = Adam(lr=config.learning_rate)
    network.compile(loss='sparse_categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
    models.append(network)

## image inference
test_label = ['toxic', 'sexual']#, 'normal']
pred_test = np.zeros((len(y_test), 3), dtype=np.float32)
for l in test_label:
    test_image_idx = [i for i in range(len(y_test)) if (y_test[i] == config.level_label_dict[l])]
    test_images = test_image_files[test_image_idx]

    for fold in range(kfold):
        pred_test[test_image_idx] += models[fold].predict(X_test[test_image_idx])
    pred_test[test_image_idx] /= kfold

## compute monitor metrics
#thresholds = {'toxic': 0.869, 'sexual': 0.561, 'normal': 0.0}
thresholds = {'toxic': 0.7, 'sexual': 0.5, 'normal': 0.0}
print('\n ====================================')
for l in test_label:
    num_true_pos, num_pred_pos, tpr, precision, recall = zz_metric(y_test, pred_test, l, thresholds[l])
    print('%s on Test: tpr %.6f, truth positve %s, predict positive %s, precision %.6f, recall %.6f.' %
        (l, tpr, num_true_pos, num_pred_pos, precision, recall))
print(' ====================================\n')

## for unknown
#err_unknown_idx = [i for i in range(len(y_test)) if(y_test[i] == 3)]
#with open('../data/raw/err_unknown.txt', 'w') as o_file:
#    for i in err_unknown_idx:
#        o_file.write('{},{:.4f},{:.4f},{:.4f}\n'.format(test_image_files[i], pred_test[i][0], pred_test[i][1], pred_test[i][2]))
#o_file.close()

## misclassified cases, debugging for low precision on toxic
pred_label = np.argmax(pred_test, axis=1)
err_normal_toxic_idx = [i for i in range(len(y_test)) if((y_test[i] == 0) & (pred_label[i] == 2))]
with open('../data/raw/err_normal_toxic.txt', 'w') as o_file:
    for i in err_normal_toxic_idx:
        o_file.write('{},{:.4f},{:.4f},{:.4f}\n'.format(test_image_files[i], pred_test[i][0], pred_test[i][1], pred_test[i][2]))
o_file.close()
err_sexual_toxic_idx = [i for i in range(len(y_test)) if((y_test[i] == 1) & (pred_label[i] == 2))]
with open('../data/raw/err_sexual_toxic.txt', 'w') as o_file:
    for i in err_sexual_toxic_idx:
        o_file.write('{},{:.4f},{:.4f},{:.4f}\n'.format(test_image_files[i], pred_test[i][0], pred_test[i][1], pred_test[i][2]))
o_file.close()

## misclassified cases, debugging for low precision on sexual
err_normal_sexual_idx = [i for i in range(len(y_test)) if((y_test[i] == 0) & (pred_label[i] == 1))]
with open('../data/raw/err_normal_sexual.txt', 'w') as o_file:
    for i in err_normal_sexual_idx:
        o_file.write('{},{:.4f},{:.4f},{:.4f}\n'.format(test_image_files[i], pred_test[i][0], pred_test[i][1], pred_test[i][2]))
o_file.close()
err_toxic_sexual_idx = [i for i in range(len(y_test)) if((y_test[i] == 2) & (pred_label[i] == 1))]
with open('../data/raw/err_toxic_sexual.txt', 'w') as o_file:
    for i in err_toxic_sexual_idx:
        o_file.write('{},{:.4f},{:.4f},{:.4f}\n'.format(test_image_files[i], pred_test[i][0], pred_test[i][1], pred_test[i][2]))
o_file.close()

## misclassified cases, debugging for normal
err_toxic_normal_idx = [i for i in range(len(y_test)) if((y_test[i] == 2) & (pred_label[i] == 0))]
with open('../data/raw/err_toxic_normal.txt', 'w') as o_file:
    for i in err_toxic_normal_idx:
        o_file.write('{},{:.4f},{:.4f},{:.4f}\n'.format(test_image_files[i], pred_test[i][0], pred_test[i][1], pred_test[i][2]))
o_file.close()

err_sexual_normal_idx = [i for i in range(len(y_test)) if((y_test[i] == 1) & (pred_label[i] == 0))]
with open('../data/raw/err_sexual_normal.txt', 'w') as o_file:
    for i in err_sexual_normal_idx:
        o_file.write('{},{:.4f},{:.4f},{:.4f}\n'.format(test_image_files[i], pred_test[i][0], pred_test[i][1], pred_test[i][2]))
o_file.close()
