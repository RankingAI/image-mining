# Created by yuanpingzhou at 11/1/18

import os
import sys
import argparse
import numpy as np
import base64
from sklearn.utils import shuffle
import time

import tensorflow as tf

from model_new_2 import OpenNSFW, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader

import utils
import config
import data_utils

IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"

train_data_source = 'history'
train_data_source_supplement = 'history_supplement'
test_data_source = '0819_new'
#strategy = 'zz_nsfw'
sample_rate = 0.1

def evaluate(sess, network, X_data, y_data, batch_size):
    num_examples = len(X_data)
    total_accuracy = 0
    total_recall = 0
    total_toxic = 0
    total_precision = 0
    total_predict = 0
    total_loss = 0.0
    sess.run(tf.local_variables_initializer())
    for offset in range(0, num_examples, batch_size):

        batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]

        loss, accuracy, recall, label_sum, precision, predict_sum = sess.run([network.loss, network.accuracy_operation, network.toxic_recall, network.toxic_label_sum, network.toxic_precision, network.toxic_predict_sum],
                                                                             feed_dict= {network.input_tensor: batch_x, network.y: batch_y, network.training: False})
        total_loss += (loss* len(batch_x))
        total_accuracy += (accuracy * len(batch_x))
        total_recall += (recall[1] * label_sum)
        total_toxic += label_sum
        total_precision += (precision[1] * predict_sum)
        total_predict += predict_sum

    return total_loss / num_examples, total_accuracy / num_examples, total_recall / total_toxic, total_precision / total_predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', "--target",
                        help="output directory",
                        default= '../data/model/nsfw/savedmodel',
                        )

    parser.add_argument("-v", "--export_version",
                        help="export model version",
                        default="2"
                        )

    parser.add_argument("-m", "--model_weights",
                        help="Path to trained model weights file",
                        default= '../data/model/nsfw/weight/open_nsfw-weights.npy',
                        )

    parser.add_argument("-t", "--input_type",
                        help="input type",
                        default=InputType.TENSOR.name.lower(),
                        #default= InputType.BASE64_JPEG.name.lower(),
                        choices=[InputType.TENSOR.name.lower(),InputType.BASE64_JPEG.name.lower()]
                        )

    parser.add_argument('-phase', "--phase",
                        default= 'train',
                        help= "project phase")

    parser.add_argument('-train_input', "--train_input",
                        default= config.data_set_route[train_data_source],
                        help="Path to the train input image. Only jpeg images are supported.")

    parser.add_argument('-test_input', "--test_input",
                        default= config.data_set_route[test_data_source],
                        help="Path to the test input image. Only jpeg images are supported.")

    parser.add_argument("-image_loader", "--image_loader",
                        default=IMAGE_LOADER_YAHOO,
                        help="image loading mechanism",
                        choices=[IMAGE_LOADER_YAHOO, IMAGE_LOADER_TENSORFLOW])

    parser.add_argument('-has_supplement', "--has_supplement",
                        default= True,
                        help= "supplement")

    args = parser.parse_args()

    # load train data set
    with utils.timer('Load image files'):
        if(args.phase == 'train'):
            image_files, labels = data_utils.load_files(args.train_input, train_data_source, sample_rate)
        if(args.has_supplement):
            image_files_supplement, labels_supplement = data_utils.load_files('%s_supplement' % args.train_input,
                                                                              train_data_source_supplement,
                                                                              sample_rate)
            print('before supplement %s' % len(image_files))
            #image_files.extend(image_files_supplement)
            image_files = np.concatenate([image_files, image_files_supplement], axis= 0)
            print('after supplement %s' % len(image_files))
            #labels.extend(labels_supplement)
            labels = np.concatenate([labels, labels_supplement], axis= 0)
        print('image files %s' % len(image_files))


    with tf.Session() as sess:
        input_type = InputType[args.input_type.upper()]
        network = OpenNSFW(weights_path= args.model_weights, num_classes= config.num_class)
        network.build(input_type= input_type)

        # function of loading image
        fn_load_image = None
        if input_type == InputType.TENSOR:
            if args.image_loader == IMAGE_LOADER_TENSORFLOW:
                fn_load_image = create_tensorflow_image_loader(sess)
            else:
                fn_load_image = create_yahoo_image_loader()
        elif input_type == InputType.BASE64_JPEG:
            fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])

        X = np.array([fn_load_image(image_files[i]).tolist() for i in range(len(image_files))])
        y = labels

        print(X.shape)
        print(y.shape)
        print('\n')

        X, y = shuffle(X, y)
        print('shuffle done!\n')
        total = X.shape[0]

        X_train, y_train = X[:int(total * 0.8),:], y[:int(total * 0.8)]
        X_valid, y_valid = X[int(total * 0.8):,:], y[int(total * 0.8):]

        num_train = X_train.shape[0]
        num_valid = X_valid.shape[0]

        print('\n=================================')
        print('total instances {}, train {}, valid {}'.format(total, num_train, num_valid))
        print('toxic {}, sexual {}, normal {}'.format(np.sum(y_train == 2), np.sum(y_train == 1), np.sum(y_train == 0)))
        print('toxic {}, sexual {}, normal {}'.format(np.sum(y_valid == 2), np.sum(y_valid == 1), np.sum(y_valid == 0)))
        print('=================================\n')

        sess.run(tf.global_variables_initializer())

        print("Training...")
        print()
        learning_rate = 0.002
        train_loss = 0.0
        train_examples = 0
        train_acc = 0.0
        start = time.time()
        for i in range(config.epochs):
            X_train, y_train = shuffle(X_train, y_train)
            if(i % 20 == 0):
                learning_rate *= pow(0.5, int(i/10))
            for offset in range(0, num_train, config.batch_size):
                end = offset + config.batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                _, loss, acc = sess.run([network.optimizer, network.loss, network.accuracy_operation], feed_dict={network.input_tensor: batch_x, network.y: batch_y, network.training: True, network.learning_rate: learning_rate})
                train_loss += (loss * len(X_train))
                train_acc += (acc * len(X_train))
                train_examples += len(X_train)

            cur = time.time()
            valid_loss, valid_accuracy, valid_precision, valid_recall = evaluate(sess, network, X_valid, y_valid, config.batch_size)
            print("EPOCH {}, time elapsed {:.2}min ...".format((i + 1), ((cur - start) / 60)))
            print('Train : loss = {:.6f}, accuracy = {:.4f}'.format((train_loss / train_examples), (train_acc / train_examples)))
            print("Valid : loss = {:.6f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}".format(valid_loss, valid_accuracy, valid_precision, valid_recall))
            print()
