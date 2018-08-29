import tensorflow as tf
import argparse
import numpy as np
import base64
import os, sys, gc

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import Callback
import keras.backend as K

from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold

from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader

import config
import utils
import data_utils

IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"

level = 'toxic'
data_source = 'history'
strategy = 'zz_nsfw'
sample_rate = 0.02

## hold the resources in the first place
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=False)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)

def extract_nsfw_features(labeled_image_root_dir, image_input_type, image_loader_type, model_dir):
    # load train data set
    with utils.timer('Load image files'):
        image_files, labels = data_utils.load_files(labeled_image_root_dir, data_source, sample_rate)
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

        # extract projection features
        with utils.timer('Projection with batching'):
            start = 0
            end = start + config.batch_size
            while (start < len(image_files)):
                if (end > len(image_files)):
                    end = len(image_files)
                with utils.timer('batch(%s) prediction' % config.batch_size):
                    batch_images = np.array([fn_load_image(image_files[i]).tolist() for i in range(start, end)])
                    X_train.extend(sess.run(projected_features, feed_dict={input_image: batch_images}).tolist())
                    y_train.extend(labels[start: end])
                print('projection %s done.' % end)
                start = end
                end = start + config.batch_size
                del batch_images
                gc.collect()
    sess.close()

    # sanity check
    assert len(y_train) == len(labels)

    return np.array(X_train), np.array(y_train)

def zz_nsfw_network():
    ''''''
    input = Input(shape= (1024, ), dtype= tf.float32, name= 'input')
    x = Dense(128, activation='relu', name= 'dense')(input)
    output_proba = Dense(config.num_class, activation='softmax', name= 'output_proba')(x)
    network = Model(input= input, output= output_proba)

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

def train(X, y):
    ''''''
    train_pred = np.zeros((len(X), config.num_class), dtype= np.float32)
    kf = StratifiedKFold(n_splits= config.kfold, shuffle= True, random_state= config.kfold_seed)
    for fold, (train_index, valid_index) in enumerate(kf.split(X, y)):
        X_train, X_valid = X[train_index], X[valid_index]
        #y_train, y_valid = ohe_y(y[train_index]), ohe_y(y[valid_index])
        y_train, y_valid = y[train_index], y[valid_index]

        # model
        network = zz_nsfw_network()
        network.compile(loss= 'sparse_categorical_crossentropy', optimizer= 'adam',metrics= ['accuracy'])

        # train
        eval = Evaluation(validation_data=(X_valid, y_valid), interval=1)
        network.fit(X_train, y_train,
                  batch_size = config.batch_size,
                  epochs= config.epochs,
                  validation_data=(X_valid, y_valid),
                  callbacks=[eval], verbose=2)

        # infer
        valid_pred_proba = network.predict(X_valid, batch_size= config.batch_size)
        train_pred[valid_index] = valid_pred_proba
    train_pred = np.argmax(train_pred, axis= 1)

    # evaluation with entire data set
    num_pred_pos = np.sum((train_pred == config.level_label_dict[level]).astype(np.int32))
    num_true_pos = np.sum((y == config.level_label_dict[level]).astype(np.int32))
    pred_label = [1 if (v >= config.level_label_dict[level]) else 0 for v in train_pred]
    truth_label = [1 if (v >= config.level_label_dict[level]) else 0 for v in y]
    cv_precision = precision_score(truth_label, pred_label)
    cv_recall = recall_score(truth_label, pred_label)
    print('\n======= Summary =======')
    print('%s-fold CV: true positive %s, predict positive %s, precision %.6f, recall %.6f' % (config.kfold, num_true_pos, num_pred_pos, cv_precision, cv_recall))
    print('=========================\n')

if __name__ == '__main__':
    ''''''
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', "--input_dir",
                        default= config.test_data_set[data_source],
                        help="Path to the input image. Only jpeg images are supported.")

    parser.add_argument('-m', "--model",
                        help="model directory",
                        default= '../data/model/nsfw/savedmodel',
                        )

    parser.add_argument('-v', '--version',
                        help= 'model version',
                        default= '1'
                        )

    parser.add_argument("-l", "--image_loader",
                        default=IMAGE_LOADER_YAHOO,
                        help="image loading mechanism",
                        choices=[IMAGE_LOADER_YAHOO, IMAGE_LOADER_TENSORFLOW])

    parser.add_argument("-t", "--input_type",
                        help="input type",
                        default=InputType.TENSOR.name.lower(),
                        #default= InputType.BASE64_JPEG.name.lower(),
                        choices=[InputType.TENSOR.name.lower(),InputType.BASE64_JPEG.name.lower()]
                        )

    args = parser.parse_args()

    X_train, y_train = extract_nsfw_features(args.input_dir, args.input_type, args.image_loader, '%s/%s' % (args.model, args.version))

    train(X_train, y_train)
