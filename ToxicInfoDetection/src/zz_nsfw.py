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
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
# tensorflow
import tensorflow as tf
from tensorflow.python.util import compat
# keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.models import model_from_json
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
train_data_source = 'history'
train_data_source_supplement = 'history_supplement'
test_data_source = '0819_new'
#strategy = 'zz_nsfw'
sample_rate = 1.0

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
            image_files_supplement, labels_supplement = data_utils.load_files('%s_supplement' % labeled_image_root_dir,
                                                                              train_data_source_supplement,
                                                                              sample_rate)
            print('before supplement %s' % len(image_files))
            #image_files.extend(image_files_supplement)
            image_files = np.concatenate([image_files, image_files_supplement], axis= 0)
            print('after supplement %s' % len(image_files))
            #labels.extend(labels_supplement)
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
    output_proba = Dense(config.num_class, activation='softmax', name= 'output_proba')(x)
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

def train(X, y, ModelWeightDir, ckptdir):
    ''''''
    # model
    network = zz_nsfw_network(print_network= False)
    network.compile(loss= 'sparse_categorical_crossentropy', optimizer= 'adam',metrics= ['accuracy'])
    #
    train_pred = np.zeros((len(X), config.num_class), dtype= np.float32)
    kf = StratifiedKFold(n_splits= config.kfold, shuffle= True, random_state= config.kfold_seed)
    for fold, (train_index, valid_index) in enumerate(kf.split(X, y)):
        X_train, X_valid = X[train_index], X[valid_index]
        #y_train, y_valid = ohe_y(y[train_index]), ohe_y(y[valid_index])
        y_train, y_valid = y[train_index], y[valid_index]

        # early stoppping
        early_stopping = EarlyStopping(monitor='val_loss', patience= 20, verbose= 10)
        # checkpoint
        #model_checkpoint = ModelCheckpoint('%s/%s.weight.%s' % (ModelWeightDir, config.strategy, fold), save_best_only=True, verbose=1)
        # custom evaluation
        evaluation = Evaluation(validation_data=(X_valid, y_valid), interval= 10)
        network.fit(X_train, y_train,
                  batch_size = config.batch_size,
                  epochs= config.epochs,
                  validation_data=(X_valid, y_valid),
                  callbacks=[evaluation, early_stopping], verbose=2)

        # infer
        valid_pred_proba = network.predict(X_valid, batch_size= config.batch_size)
        train_pred[valid_index] = valid_pred_proba

    SaveCheckpoint(network, ckptdir)

    # evaluation with entire data set
    cv_num_true_pos, cv_num_pred_pos, cv_accuracy, cv_precision, cv_recall = zz_metric(y, train_pred, 'toxic')
    print('\n======= Summary =======')
    print('%s-fold CV: true positive %s, predict positive %s, accuracy %.6f, precision %.6f, recall %.6f' %
          (config.kfold, cv_num_true_pos, cv_num_pred_pos, cv_accuracy, cv_precision, cv_recall))
    print('=========================\n')

def zz_metric(y, predict, level):
    ''''''
    # convert probabilities to label sequence number
    pred_label = np.argmax(predict, axis=1)
    # number of the positives
    num_pred_pos = np.sum((pred_label == config.level_label_dict[level]).astype(np.int32))
    num_true_pos = np.sum((y == config.level_label_dict[level]).astype(np.int32))
    # convert into binary mode
    pred_label = [1 if (v == config.level_label_dict[level]) else 0 for v in pred_label]
    truth_label = [1 if (v == config.level_label_dict[level]) else 0 for v in y]
    # precision/recall/accuracy
    accuracy = np.sum((np.array(pred_label) == np.array(truth_label)).astype(np.int32))/len(pred_label)
    precision = precision_score(truth_label, pred_label)
    recall = recall_score(truth_label, pred_label)

    return num_true_pos, num_pred_pos, accuracy, precision, recall

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
    for l in ['toxic', 'sexual', 'normal']:
        num_true_pos, num_pred_pos, accuracy, precision, recall = zz_metric(y, pred_test, l)
        print('%s on Test: accuracy %.6f, truth positve %s, predict positive %s, precision %.6f, recall %.6f.' %
            (l, accuracy, num_true_pos, num_pred_pos, precision, recall))
    print('==========================\n')

if __name__ == '__main__':
    ''''''
    parser = argparse.ArgumentParser()

    parser.add_argument('-phase', "--phase",
                        default= 'test',
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
    # for inferring
    inferdir = '%s/%s/infer' % (config.ModelRootDir, config.strategy)
    if(os.path.exists(inferdir) == False):
        os.makedirs(inferdir)
    # for testing
    testdir = '%s/%s/test' % (config.ModelRootDir, config.strategy)
    if(os.path.exists(testdir) == False):
        os.makedirs(testdir)

    if(args.phase == 'train'):
        X_train, y_train = extract_nsfw_features(args.train_input,
                                                 args.image_data_type,
                                                 args.image_loader,
                                                 '%s/%s' % (args.nsfw_model, args.model_version),
                                                 has_supplement= True,
                                                 phase= args.phase,
                                                 return_image_files= False)
        train(X_train, y_train, ModelWeightDir, ckptdir)
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
