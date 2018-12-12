# Created by yuanpingzhou at 12/3/18

import urllib.request
import numpy as np
import skimage
import skimage.io
from PIL import Image
from io import BytesIO
import datetime
import time
import logging

import tensorflow as tf

# grpc
import ToxicImageDetection_pb2
import ToxicImageDetection_pb2_grpc

import utils

class ModelFactory(object):

    ##
    nsfw_input_tensor_name = 'input:0'
    nsfw_output_tensor_name = 'nsfw_features:0'
    nsfw_graph = None
    nsfw_sess = None
    nsfw_input_tensor = None
    nsfw_output_tensor = None
    ##
    zz_nsfw_input_tensor_name = 'input:0'
    zz_nsfw_output_tensor_name = 'output_proba/Softmax:0'
    zz_nsfw_graph = []
    zz_nsfw_sess = []
    zz_nsfw_input_tensor = []
    zz_nsfw_output_tensor = []

    @staticmethod
    def load_saved_model(saved_model_dir, input_tensor_name, output_tensor_name):
        ''''''
        graph = tf.Graph()
        sess = tf.Session(graph= graph, config= tf.ConfigProto(use_per_session_threads= False))
        # load model
        with utils.timer('Load model {}'.format(saved_model_dir)):
            tf.saved_model.loader.load(sess, ["serve"], saved_model_dir)
            # extract tensor from graph
            input_tensor = graph.get_tensor_by_name(input_tensor_name)
            output_tensor = graph.get_tensor_by_name(output_tensor_name)

        logging.debug('load model {} done'.format(saved_model_dir))

        return graph, sess, input_tensor, output_tensor

    @staticmethod
    def initial_models(num_zz_nsfw_model, nsfw_saved_model_dir, nsfw_model_version, zz_nsfw_saved_model_dir, zz_nsfw_model_version):
        ModelFactory.nsfw_graph, ModelFactory.nsfw_sess, \
        ModelFactory.nsfw_input_tensor, ModelFactory.nsfw_output_tensor = \
            ModelFactory.load_saved_model(
            '{}/{}'.format(nsfw_saved_model_dir, nsfw_model_version),
            ModelFactory.nsfw_input_tensor_name,
            ModelFactory.nsfw_output_tensor_name
        )
        for fold in range(num_zz_nsfw_model):
            saved_model_dir = '{}/{}/{}'.format(zz_nsfw_saved_model_dir, fold, zz_nsfw_model_version)
            graph, sess, input_tensor, output_tensor = ModelFactory.load_saved_model(saved_model_dir,
                                                                                     ModelFactory.zz_nsfw_input_tensor_name,
                                                                                     ModelFactory.zz_nsfw_output_tensor_name)
            ModelFactory.zz_nsfw_graph.append(graph)
            ModelFactory.zz_nsfw_sess.append(sess)
            ModelFactory.zz_nsfw_input_tensor.append(input_tensor)
            ModelFactory.zz_nsfw_output_tensor.append(output_tensor)

# create a class to define the server functions
class ImageClassifyService(ToxicImageDetection_pb2_grpc.NSFWServicer):

    def __init__(self, num_zz_nsfw_model= 5, num_class= 3, image_size= -1):
        self.num_class = num_class
        self.num_zz_nsfw_model = num_zz_nsfw_model
        self.image_size = image_size
        logging.debug('initial ImageClassifyService: num_zz_nsfw_model {}, num_class {}, image_size {}'.format(num_zz_nsfw_model, num_class, image_size))

    def download_image(self, image_url):
        ''''''
        rsp = urllib.request.urlopen(image_url)
        return rsp.read()

    def preprocessing_image(self, image):
        ''''''
        VGG_MEAN = [104, 117, 123]

        #pimg = image
        img_data = image
        im = Image.open(BytesIO(img_data))

        if im.mode != "RGB":
            im = im.convert('RGB')

        imr = im.resize((256, 256), resample=Image.BILINEAR)

        fh_im = BytesIO()
        imr.save(fh_im, format='JPEG')
        fh_im.seek(0)

        image = (skimage.img_as_float(skimage.io.imread(fh_im, as_grey=False)).astype(np.float32))

        H, W, _ = image.shape
        h, w = (224, 224)

        h_off = max((H - h) // 2, 0)
        w_off = max((W - w) // 2, 0)
        image = image[h_off:h_off + h, w_off:w_off + w, :]

        # RGB to BGR
        image = image[:, :, :: -1]

        image = image.astype(np.float32, copy=False)
        image = image * 255.0
        image -= np.array(VGG_MEAN, dtype=np.float32)

        return image

    def extract_nsfw_features_multiple(self, images):
        ''''''
        with ModelFactory.nsfw_sess.graph.as_default():
            result = ModelFactory.nsfw_sess.run(ModelFactory.nsfw_output_tensor, feed_dict= {ModelFactory.nsfw_input_tensor: images})
        return result

    def predict(self, feats):
        ''''''
        results = np.zeros((feats.shape[0], self.num_class))
        for fold in range(self.num_zz_nsfw_model):
            with ModelFactory.zz_nsfw_sess[fold].graph.as_default():
                pred = ModelFactory.zz_nsfw_sess[fold].run(ModelFactory.zz_nsfw_output_tensor[fold], feed_dict= {ModelFactory.zz_nsfw_input_tensor[fold]: feats})
            results += pred
        return results / self.num_zz_nsfw_model

    def download_all(self, urls):
        ''''''
        images = []
        for url in urls:
            try:
                if(self.image_size >= 0):
                    img = self.download_image('{}?w={}'.format(url, self.image_size))
                else:
                    img = self.download_image(url)
            except Exception as e:
                img = None
                logging.debug('download image {} exception '.format(urls[i]))
            images.append(img)
        return images

    def preprocess_all(self, urls, images):
        ''''''
        new_images = []
        for i in range(len(images)):
            try:
                img = self.preprocessing_image(images[i])
            except Exception as e:
                img = np.zeros((224, 224, 3), dtype=np.float32)
                logging.debug('preprocess image {} exception '.format(urls[i]))
            new_images.append(img.tolist())
        return np.array(new_images)

    def extract_nsfw_feature_all(self, images):
        ''''''
        return self.extract_nsfw_features_multiple(images)

    def predict_all(self, features):
        ''''''
        return self.predict(features)

    def pack_all(self, predicts):
        ''''''
        response = ToxicImageDetection_pb2.NSFWResult()
        results = []
        for r in range(predicts.shape[0]):
            p = ToxicImageDetection_pb2.Probabilities()
            p.normal = predicts[r][0]
            p.sexual = predicts[r][1]
            p.toxic = predicts[r][2]
            results.append(p)
        response.result.extend(results)

        return response

    def OpenNSFW(self, request, context):
        ''''''
        t1 = time.time()
        image_bytes = self.download_all(request.urls)
        t2 = time.time()
        images = self.preprocess_all(request.urls, image_bytes)
        t3 = time.time()
        features = self.extract_nsfw_feature_all(images)
        t4 = time.time()
        predicts = self.predict_all(features)
        t5 = time.time()
        response = self.pack_all(predicts)
        t6 = time.time()

        print('download {:.2f}, preprocess {:.2f}, nsfw_feature {:.2f}, predict {:.2f}, pack {:.2f}'.format((t2 - t1),
                                                                                                            (t3 - t2),
                                                                                                            (t4 - t3),
                                                                                                            (t5 - t4),
                                                                                                            (t6 - t5)
                                                                                                            ))

        return response
