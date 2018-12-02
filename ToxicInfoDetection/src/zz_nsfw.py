####################################################################################
# This is a pipeline of toxic image detection applied model transferred from nsfw. #
# Updated by yuanpingzhou on 8/30/2018                                             #
####################################################################################
import argparse
import numpy as np
import base64
import os, sys, gc
import shutil
# sklearn
from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold
# tensorflow
import tensorflow as tf
from tensorflow.python.util import compat
# keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras.backend as K
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
# custom
from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader
import config
import utils
import data_utils

IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"

level = 'toxic'
#train_data_source = 'history'
train_data_source = '1109'
train_data_source_supplement = 'history_supplement'
#test_data_source = '0819_new'
test_data_source = 'test_0819_part1'
#strategy = 'zz_nsfw'
sample_rate = 1.00

## hold the resources in the first place
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=False)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)

def SaveCheckpoint(model, outputdir):
    ''''''
    with utils.timer('save ckpt model'):
        # save model schema
        model_json = model.to_json()
        with open("%s/%s.json" % (outputdir, config.strategy), "w") as o_file:
            o_file.write(model_json)
        o_file.close()
        # save model weights
        model.save_weights("%s/%s.h5" % (outputdir, config.strategy))

def LoadCheckpoint(inputdir):
    ''''''
    with utils.timer('load ckpt model'):
        # load model schema
        with open("%s/%s.json" % (inputdir, config.strategy), "r") as i_file:
            loaded_model_json = i_file.read()
        i_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load model weights
        loaded_model.load_weights("%s/%s.h5" % (inputdir, config.strategy))

    return loaded_model

def export_model(model_dir, model_version, model):
    ''''''
    path = os.path.dirname(os.path.abspath(model_dir))
    if os.path.isdir(path) == False:
        os.makedirs(path)

    export_path = os.path.join(
        compat.as_bytes(model_dir),
        compat.as_bytes(str(model_version)))

    if os.path.isdir(export_path) == True:
        shutil.rmtree(export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    model_input = tf.saved_model.utils.build_tensor_info(model.input)
    model_output = tf.saved_model.utils.build_tensor_info(model.output)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'x': model_input},
            outputs={'probabilities': model_output},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    )

    legacy_init = tf.group(tf.tables_initializer(), name='legacy_init_op')

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess= sess,
            tags= [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict':
                    prediction_signature,
            }, legacy_init_op= legacy_init)

        builder.save()

def extract_nsfw_features(labeled_image_root_dir,
                          image_input_type,
                          image_loader_type,
                          model_dir,
                          has_supplement= False,
                          phase= 'train',
                          return_image_files= False):
    # load train data set
    with utils.timer('Load image files'):
        if(phase== 'train'):
            image_files, labels = data_utils.load_files(labeled_image_root_dir, train_data_source, sample_rate)
        else:
            image_files, labels = data_utils.load_files(labeled_image_root_dir, test_data_source, sample_rate)
        if(has_supplement):
            for part in ['test_0819_part1_1', 'test_0819_part1_2', 'test_0819_part1_3','test_0819_part3_1']:
                supplement_dir = '{}/{}'.format('/'.join(labeled_image_root_dir.split('/')[:-1]), part)
                image_files_supplement, labels_supplement = data_utils.load_files(supplement_dir,
                                                                              'test_0819_part1',
                                                                              sample_rate)
                print('before supplement %s' % len(image_files))
                image_files = np.concatenate([image_files, image_files_supplement], axis= 0)
                print('after supplement %s' % len(image_files))
                labels = np.concatenate([labels, labels_supplement], axis= 0)
        print('image files %s' % len(image_files))

    X_train = []
    y_train = []
    # transform original image into nsfw features
    with tf.Session(graph=tf.Graph()) as sess:

        input_type = InputType[image_input_type.upper()]

        # function of loading image
        fn_load_image = None
        if input_type == InputType.TENSOR:
            if image_loader_type == IMAGE_LOADER_TENSORFLOW:
                fn_load_image = create_tensorflow_image_loader(sess)
            else:
                fn_load_image = create_yahoo_image_loader()
        elif input_type == InputType.BASE64_JPEG:
            fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])

        # load model
        with utils.timer('Load model'):
            tf.saved_model.loader.load(sess, ["serve"], model_dir)
            graph = tf.get_default_graph()
            # extract tensor from graph
            input_image = graph.get_tensor_by_name("input:0")
            projected_features = graph.get_tensor_by_name('nsfw_features:0')
            predict_proba = graph.get_tensor_by_name("predictions:0")

        nsfw_batch_size = 512
        # extract projection features
        with utils.timer('Projection with batching'):
            start = 0
            end = start + nsfw_batch_size
            while (start < len(image_files)):
                if (end > len(image_files)):
                    end = len(image_files)
                with utils.timer('batch(%s) prediction' % nsfw_batch_size):
                    batch_images = np.array([fn_load_image(image_files[i]).tolist() for i in range(start, end)])
                    X_train.extend(sess.run(projected_features, feed_dict={input_image: batch_images}).tolist())
                    y_train.extend(labels[start: end])
                print('projection %s done.' % end)
                start = end
                end = start + nsfw_batch_size
                del batch_images
                gc.collect()
    sess.close()

    # sanity check
    assert len(y_train) == len(labels)

    if(return_image_files == True):
        return np.array(X_train), np.array(y_train), image_files
    else:
        return np.array(X_train), np.array(y_train)

def zz_nsfw_network(print_network= True):
    ''''''
    input = Input(shape= (1024, ), name= 'input')
    x = Dense(256, activation='relu', name= 'dense_0')(input)
    #x = Dropout(0.25)(x)
    #x = Dense(256, activation='relu', name= 'dense_1')(x)
    #x = Dropout(0.25)(x)
    output_proba = Dense(3, activation='softmax', name= 'output_proba')(x)
    network = Model(input= input, output= output_proba)

    if(print_network == True):
        network.summary()

    return network

class Evaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = np.argmax(self.model.predict(self.X_val, verbose= 0), axis= 1)
            #y_pred = self.model.predict(self.X_val, verbose= 0)
            num_pred_pos = np.sum((y_pred == config.level_label_dict[level]).astype(np.int32))
            pred_label = [1 if(v >= config.level_label_dict[level]) else 0 for v in y_pred]
            truth_label = [1 if(v >= config.level_label_dict[level]) else 0 for v in self.y_val]
            precision = precision_score(truth_label, pred_label)
            recall = recall_score(truth_label, pred_label)
            print("\n epoch: %d - %s positive %s - precision %.6f - recall %.6f\n" % (epoch + 1, level, num_pred_pos, precision, recall))

def ohe_y(labels):
    ''''''
    ohe = np.zeros((len(labels), config.num_class), dtype=np.float32)
    ohe[np.arange(config.num_class), labels] = 1.0

    return ohe

def debug_block(pred, truth, debug_root_dir, image_files):
    pred_label = np.argmax(pred, axis=1)
    ## misclassified cases, debugging for low precision on toxic
    err_normal_toxic_idx = [i for i in range(len(truth)) if ((truth[i] == 0) & (pred_label[i] == 2))]
    with open('{}/err_normal_toxic.txt'.format(debug_root_dir), 'w') as o_file:
        for i in err_normal_toxic_idx:
            o_file.write('{},{:.4f},{:.4f},{:.4f}\n'.format(image_files[i], pred[i][0], pred[i][1],
                                                            pred[i][2]))
    o_file.close()
    err_sexual_toxic_idx = [i for i in range(len(truth)) if ((truth[i] == 1) & (pred_label[i] == 2))]
    with open('{}/err_sexual_toxic.txt'.format(debug_root_dir), 'w') as o_file:
        for i in err_sexual_toxic_idx:
            o_file.write('{},{:.4f},{:.4f},{:.4f}\n'.format(image_files[i], pred[i][0], pred[i][1],
                                                            pred[i][2]))
    o_file.close()

    ## misclassified cases, debugging for low precision on sexual
    err_normal_sexual_idx = [i for i in range(len(truth)) if ((truth[i] == 0) & (pred_label[i] == 1))]
    with open('{}/err_normal_sexual.txt'.format(debug_root_dir), 'w') as o_file:
        for i in err_normal_sexual_idx:
            o_file.write('{},{:.4f},{:.4f},{:.4f}\n'.format(image_files[i], pred[i][0], pred[i][1],
                                                            pred[i][2]))
    o_file.close()
    err_toxic_sexual_idx = [i for i in range(len(truth)) if ((truth[i] == 2) & (pred_label[i] == 1))]
    with open('{}/err_toxic_sexual.txt'.format(debug_root_dir), 'w') as o_file:
        for i in err_toxic_sexual_idx:
            o_file.write('{},{:.4f},{:.4f},{:.4f}\n'.format(image_files[i], pred[i][0], pred[i][1],
                                                            pred[i][2]))
    o_file.close()

    ## misclassified cases, debugging for normal
    err_toxic_normal_idx = [i for i in range(len(truth)) if ((truth[i] == 2) & (pred_label[i] == 0))]
    with open('{}/err_toxic_normal.txt'.format(debug_root_dir), 'w') as o_file:
        for i in err_toxic_normal_idx:
            o_file.write('{},{:.4f},{:.4f},{:.4f}\n'.format(image_files[i], pred[i][0], pred[i][1],
                                                            pred[i][2]))
    o_file.close()

    err_sexual_normal_idx = [i for i in range(len(truth)) if ((truth[i] == 1) & (pred_label[i] == 0))]
    with open('{}/err_sexual_normal.txt'.format(debug_root_dir), 'w') as o_file:
        for i in err_sexual_normal_idx:
            o_file.write('{},{:.4f},{:.4f},{:.4f}\n'.format(image_files[i], pred[i][0], pred[i][1],
                                                            pred[i][2]))
    o_file.close()

def train(X, y, ModelWeightDir, GraphDir, train_image_files):
    ''''''
    train_pred = np.zeros((len(X), config.num_class), dtype= np.float32)
    kf = StratifiedKFold(n_splits= config.kfold, shuffle= True, random_state= config.kfold_seed)
    for fold, (train_index, valid_index) in enumerate(kf.split(X, y)):
        print('\n ========================== FOLD {}'.format(fold))
        # model
        network = zz_nsfw_network(print_network= False)
        opti = Adam(lr= config.learning_rate)
        network.compile(loss= 'sparse_categorical_crossentropy', optimizer= opti, metrics= ['accuracy'])

        X_train, X_valid = X[train_index], X[valid_index]
        #y_train, y_valid = ohe_y(y[train_index]), ohe_y(y[valid_index])
        y_train, y_valid = y[train_index], y[valid_index]

        # early stoppping
        early_stopping = EarlyStopping(monitor='val_loss', mode= 'min', patience= 18, verbose= 1)
        # checkpoint
        model_checkpoint = ModelCheckpoint('%s/%s.weight.%s' % (ModelWeightDir, config.strategy, fold), monitor= 'val_loss', mode= 'min', save_best_only=True, verbose=1)
        # learning rate schedual
        lr_schedule = ReduceLROnPlateau(monitor= 'val_loss', mode= 'min', factor= 0.5, patience= 6, min_lr= 0.00001,verbose=1)
        # custom evaluation
        evaluation = Evaluation(validation_data=(X_valid, y_valid), interval= 1)
        # tensorboard
        tbCallBack = TensorBoard(log_dir='{}/{}'.format(GraphDir, fold), histogram_freq=0, write_graph=True, write_images=True)

        network.fit(X_train, y_train,
                  batch_size = config.batch_size,
                  epochs= config.epochs,
                  validation_data=(X_valid, y_valid),
                  callbacks=[evaluation, early_stopping, model_checkpoint, lr_schedule, tbCallBack],
                  verbose=2)

        # infer
        valid_pred_proba = network.predict(X_valid, batch_size= config.batch_size)
        train_pred[valid_index] = valid_pred_proba

    #SaveCheckpoint(network, ckptdir)
    debug_root_dir = '{}/{}/debug'.format(config.ModelRootDir, config.strategy)
    if(os.path.exists(debug_root_dir) == False):
        os.makedirs(debug_root_dir)
    debug_block(train_pred, y, debug_root_dir, train_image_files)

    # evaluation with entire data set
    print('\n======= Summary =======')
    for l in ['toxic', 'sexual']:#, 'normal']:
        cv_num_true_pos, cv_num_pred_pos, best_threshold, best_f1, best_precision, best_recall, tpr = zz_metric_threshold(y, train_pred[:, config.level_label_dict[l]], l)
        print('%s-fold CV for %s: true positive %s, predict positive %s, best threshold %.6f, f1 %.6f, precision %.6f, recall %.6f, tpr %.6f' %
            (config.kfold, l, cv_num_true_pos, cv_num_pred_pos, best_threshold, best_f1, best_precision, best_recall, tpr))
    print('=========================\n')

def best_threshold_based_f1(y, predict, level):
    ''''''
    possible_thresholds = np.linspace(0.4, 0.9, 60)
    f1_scores = np.zeros(len(possible_thresholds))
    recalls = np.zeros(len(possible_thresholds))
    precisions = np.zeros(len(possible_thresholds))
    for i in range(len(possible_thresholds)):
        threshold = possible_thresholds[i]
        pred_label = [1 if(v > threshold) else 0 for v in predict]
        truth_label = [1 if (v == config.level_label_dict[level]) else 0 for v in y]
        precisions[i] = precision_score(truth_label, pred_label)
        recalls[i] = recall_score(truth_label, pred_label)
        f1_scores[i] = (2 * precisions[i] * recalls[i])/(3 * precisions[i] + 5 * recalls[i])
    best_idx = np.argmax(f1_scores)

    return possible_thresholds[best_idx], f1_scores[best_idx], recalls[best_idx], precisions[best_idx]

def zz_metric_threshold(y, predict, level):
    ''''''
    best_threshold, best_f1, best_recall, best_precision = best_threshold_based_f1(y, predict, level)
    pred_label = [1 if(v > best_threshold) else 0 for v in predict]
    truth_label = [1 if (v == config.level_label_dict[level]) else 0 for v in y]
    #best_accuracy = np.sum((np.array(pred_label) == np.array(truth_label)).astype(np.int32)) / len(pred_label)
    tpr = sum_weighted_tpr(y, predict)

    return np.sum(truth_label), np.sum(pred_label), best_threshold, best_f1, best_precision, best_recall, tpr

## metric function, sum of weighted TPR
def sum_weighted_tpr(y, scores):
    ''''''
    swt = .0
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label= 1)
    for t in config.tpr_factor.keys():
        swt += config.tpr_factor[t] * tpr[np.where(fpr >= t)[0][0]]
    return swt

def zz_metric(y, predict, level, threshold):
    ''''''
    # convert probabilities to label sequence number
    pred_max_idx = np.argmax(predict, axis=1)
    if((level == 'toxic') | (level == 'sexual')):
        pred_label = np.array([pred_max_idx[i] if(predict[i][pred_max_idx[i]] > threshold) else -1 for i in range(len(predict))])
    else:
        pred_label = pred_max_idx
    # number of the positives
    num_pred_pos = np.sum((pred_label == config.level_label_dict[level]).astype(np.int32))
    num_true_pos = np.sum((y == config.level_label_dict[level]).astype(np.int32))
    # convert into binary mode
    pred_label = [1 if (v == config.level_label_dict[level]) else 0 for v in pred_label]
    truth_label = [1 if (v == config.level_label_dict[level]) else 0 for v in y]
    # precision/recall/accuracy
    #accuracy = np.sum((np.array(pred_label) == np.array(truth_label)).astype(np.int32))/len(pred_label)
    precision = precision_score(truth_label, pred_label)
    recall = recall_score(truth_label, pred_label)

    tpr = sum_weighted_tpr(y, predict[:, config.level_label_dict[level]])

    return num_true_pos, num_pred_pos, tpr, precision, recall

def test(X, y, model, image_files, testdir):
    ''''''
    assert len(image_files) == len(X)

    # infer
    model.compile(loss= 'sparse_categorical_crossentropy', optimizer= 'adam',metrics= ['accuracy'])
    pred_test = model.predict(X, batch_size= config.batch_size)
    pred_label = np.argmax(pred_test, axis=1)

    # saving
    with open('%s/test_log.txt' % testdir, 'w') as o_file:
        for i in range(len(image_files)):
            o_file.write('%s,%s,%s,%.6f,%.6f,%.6f\n' % (image_files[i], y[i], pred_label[i], pred_test[i][0], pred_test[i][1], pred_test[i][2]))
    o_file.close()

    # evaluation
    print('\n ======= Summary ========')
    for l in ['toxic', 'sexual']:#, 'normal']:
        num_true_pos, num_pred_pos, accuracy, precision, recall = zz_metric(y, pred_test, l, config.thresholds[l])
        print('%s on Test: accuracy %.6f, truth positve %s, predict positive %s, precision %.6f, recall %.6f.' %
            (l, accuracy, num_true_pos, num_pred_pos, precision, recall))
    print('==========================\n')

if __name__ == '__main__':
    ''''''
    parser = argparse.ArgumentParser()

    parser.add_argument('-phase', "--phase",
                        default= 'train',
                        help= "project phase")

    parser.add_argument('-train_input', "--train_input",
                        default= config.data_set_route[train_data_source],
                        help="Path to the train input image. Only jpeg images are supported.")

    parser.add_argument('-test_input', "--test_input",
                        default= config.data_set_route[test_data_source],
                        help="Path to the test input image. Only jpeg images are supported.")

    parser.add_argument('-nsfw_model', "--nsfw_model",
                        help="model directory",
                        default= '../data/model/nsfw/savedmodel',
                        )

    parser.add_argument('-model_version', '--model_version',
                        help= 'model version',
                        default= '1'
                        )

    parser.add_argument("-image_loader", "--image_loader",
                        default=IMAGE_LOADER_YAHOO,
                        help="image loading mechanism",
                        choices=[IMAGE_LOADER_YAHOO, IMAGE_LOADER_TENSORFLOW])

    parser.add_argument("-image_data_type", "--image_data_type",
                        help="image data type",
                        default=InputType.TENSOR.name.lower(),
                        #default= InputType.BASE64_JPEG.name.lower(),
                        choices=[InputType.TENSOR.name.lower(),InputType.BASE64_JPEG.name.lower()]
                        )

    args = parser.parse_args()

    # for training
    ModelWeightDir = '%s/%s/weight' % (config.ModelRootDir, config.strategy)
    if(os.path.exists(ModelWeightDir) == False):
        os.makedirs(ModelWeightDir)
    # for exporting
    ckptdir = '%s/%s/ckpt' % (config.ModelRootDir, config.strategy)
    if(os.path.exists(ckptdir) == False):
        os.makedirs(ckptdir)
    # for tensorboard
    tbdir = '{}/{}/graph'.format(config.ModelRootDir, config.strategy)
    if(os.path.exists(tbdir) == False):
        os.makedirs(tbdir)
    # for inferring
    inferdir = '%s/%s/infer' % (config.ModelRootDir, config.strategy)
    if(os.path.exists(inferdir) == False):
        os.makedirs(inferdir)
    # for testing
    testdir = '%s/%s/test' % (config.ModelRootDir, config.strategy)
    if(os.path.exists(testdir) == False):
        os.makedirs(testdir)

    if(args.phase == 'train'):
        X_train, y_train, train_image_files = extract_nsfw_features(args.train_input,
                                                 args.image_data_type,
                                                 args.image_loader,
                                                 '%s/%s' % (args.nsfw_model, args.model_version),
                                                 has_supplement= True,
                                                 phase= args.phase,
                                                 return_image_files= True)
        train(X_train, y_train, ModelWeightDir, tbdir, train_image_files)
    elif(args.phase == 'export'):
        K.set_learning_phase(0) ##!!! need to be set before loading model
        model = LoadCheckpoint(ckptdir)
        model.summary()
        export_model(inferdir, args.model_version, model) # share the version number with nsfw
    elif(args.phase == 'test'):
        model = LoadCheckpoint(ckptdir)
        X_test, y_test, test_image_files = extract_nsfw_features(args.test_input,
                                                                args.image_data_type,
                                                                args.image_loader,
                                                                '%s/%s' % (args.nsfw_model, args.model_version),
                                                                has_supplement= False,
                                                                phase= args.phase,
                                                                return_image_files= True)
        print(X_test.shape)
        print(y_test.shape)
        test(X_test, y_test, model, test_image_files, testdir)
