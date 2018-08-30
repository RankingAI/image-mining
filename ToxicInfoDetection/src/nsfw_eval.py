########################################################################
# This is evaluation script for NSFW model which is authored by YAHOO. #
# Update by yuanpingzhou on 8/28/2018                                  #
########################################################################

import sys, os, time
import argparse
import tensorflow as tf

from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader

import numpy as np

import config
import data_utils
import plot_utils
import utils

IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"

level = 'toxic'
data_source = '0819'
sample_rate = 1.0

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7, allow_growth=False)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', "--input_dir", default= config.test_data_set[data_source],
                        help="Path to the input image. Only jpeg images are supported.")

    parser.add_argument("-m", "--model_weights", default= config.nsfw_model_weight_file,
                        help="Path to trained model weights file")

    parser.add_argument("-l", "--image_loader",default=IMAGE_LOADER_YAHOO,
                        help="image loading mechanism",
                        choices=[IMAGE_LOADER_YAHOO, IMAGE_LOADER_TENSORFLOW])

    parser.add_argument("-t", "--input_type",default=InputType.TENSOR.name.lower(),
                        help="input type",
                        choices=[InputType.TENSOR.name.lower(),
                                 InputType.BASE64_JPEG.name.lower()])

    args = parser.parse_args()

    with utils.timer('Load image files'):
        image_files, labels = data_utils.load_files(args.input_dir, data_source, sample_rate)
        print('image files %s' % len(image_files))

    model = OpenNsfwModel()

    predictions = []

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        input_type = InputType[args.input_type.upper()]

        with utils.timer('Load model weight'):
            model.build(weights_path=args.model_weights, input_type=input_type)

        fn_load_image = None

        if input_type == InputType.TENSOR:
            if args.image_loader == IMAGE_LOADER_TENSORFLOW:
                fn_load_image = create_tensorflow_image_loader(sess)
            else:
                fn_load_image = create_yahoo_image_loader()
        elif input_type == InputType.BASE64_JPEG:
            import base64
            fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])

        sess.run(tf.global_variables_initializer())

        with utils.timer('Prediction'):
            start = 0
            end = start + config.batch_size
            while(start < len(image_files)):
                if(end > len(image_files)):
                    end = len(image_files)
                with utils.timer('Batch[%s] prediction' % config.batch_size):
                    batch_images = [fn_load_image(image_files[i]) for i in range(start, end)]
                    #print(batch_images[0].shape)
                    #sys.exit(1)
                    predictions.extend(sess.run(model.predictions, feed_dict= {model.input: batch_images})[:,1])
                print('Prediction %s done.' % end)
                start = end
                end = start + config.batch_size

    # save
    PredictOutputFile = '%s/%s.csv' % (config.TestOutputDir, data_source)
    with utils.timer('Save predictions'):
        data_utils.save_predictions(image_files, labels, predictions, PredictOutputFile)

    # visualization on threshold for f1/precision/recall
    if(data_source == 'hisotry'):
        output_image_file ='%s/%s_vs_threshold.jpg' % (config.TestOutputDir, level)
        with utils.timer('Save visualization for threshold'):
            plot_utils.threshold_vs_toxic(labels, predictions, level, output_image_file)

if __name__ == "__main__":
    main(sys.argv)
