# Created by yuanpingzhou at 11/7/18

import keras
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.layers import Activation, Add, Multiply, Input, Flatten, Dense
from keras.models import Model
import keras.backend as K
from keras.preprocessing.image import load_img
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras_metrics

import tensorflow as tf

import glob

import os,sys
from random import shuffle
import numpy as np
import config
import utils

# configuration for GPU resources
with K.tf.device('/device:GPU:0'):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 0.8, allow_growth=False)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)

class AttentionResidualNetwork():
    ''''''
    def __init__(self, shape):
        self.shape = shape

    def __residual_block(self, input_tensor, filters , kernel_size= 3, scope= ''):
        ''''''
        prefix = '{}_res'.format(scope)

        # BN
        x = BatchNormalization(name= '{}_bn_0'.format(prefix))(input_tensor)

        # Conv 1*1 & BN
        x = Conv2D(filters= filters, kernel_size= 1, strides= 1, padding= 'same', name= '{}_conv_1'.format(prefix))(x)
        x = BatchNormalization(name= '{}_bn_1'.format(prefix))(x)
        x = Activation('sigmoid', name= '{}_act_1'.format(prefix))(x)

        # Conv 3*3 & BN
        x = Conv2D(filters= filters, kernel_size= kernel_size, strides= 1, padding= 'same', name= '{}_conv_2'.format(prefix))(x)

        # coordinate the shape of the x and inpute_tensor
        if(K.shape(input_tensor)[-1] != filters):
            input_tensor = Conv2D(filters= filters, kernel_size= 1, strides= 1, padding= 'same', name= '{}_conv_0'.format(prefix))(input_tensor)

        return Add(name= '{}_add'.format(prefix))([input_tensor, x])

    def __attention_block(self, input_tensor, filters, scope= ''):
        ''''''
        prefix = '{}_att'.format(scope)

        p, t, r= [1, 2, 1]

        # preprocessing block
        for i in range(p):
            input_tensor = self.__residual_block(input_tensor, filters= filters, kernel_size= 3, scope = '{}_preprocess_{}'.format(prefix, i))

        # trunk branch
        output_trunk = input_tensor
        for i in range(t):
            output_trunk = self.__residual_block(output_trunk, filters= filters, kernel_size= 3, scope = '{}_trunk_branch_{}'.format(prefix, i))

        # soft mask branch
        output_soft_mask = input_tensor

        # down sampling 1
        output_soft_mask = MaxPooling2D(pool_size= (1, 2), strides= (1, 2), padding= 'same',
                                        name= '{}_soft_mask_branch_down_sampling1_pooling'.format(prefix))(output_soft_mask)
        for i in range(r):
            output_soft_mask = self.__residual_block(output_soft_mask, filters= filters, kernel_size= 3,
                                              scope = '{}_soft_mask_branch_down_sampling1'.format(prefix))

        output_soft_mask_skip_connect = self.__residual_block(output_soft_mask, filters= filters, kernel_size= 3,
                                                       scope = '{}_soft_mask_branch_down_sampling1_skip_connect'.format(prefix))

        # down sampling 2
        output_soft_mask = MaxPooling2D(pool_size= (1, 2), strides= (1, 2), padding= 'same',
                                        name= '{}_soft_mask_branch_down_sampling2_pooling'.format(prefix))(output_soft_mask)
        for i in range(r):
            output_soft_mask = self.__residual_block(output_soft_mask, filters= filters, kernel_size= 3,
                                              scope = '{}_soft_mask_branch_down_sampling2'.format(prefix))

        # up sampling 2 and add
        for i in range(r):
            output_soft_mask = self.__residual_block(output_soft_mask, filters= filters, kernel_size= 3,
                                              scope = '{}_soft_mask_branch_up_sampling2'.format(prefix))
        output_soft_mask = UpSampling2D(size= [1, 2], name= '{}_soft_mask_branch_up_sampling2'.format(prefix))(output_soft_mask)
        output_soft_mask = Add(name= '{}_soft_mask_branch_add2'.format(prefix))([output_soft_mask_skip_connect, output_soft_mask])

        # up sampling 1
        for i in range(r):
            output_soft_mask = self.__residual_block(output_soft_mask, filters= filters, kernel_size= 3,
                           scope = '{}_soft_mask_branch_up_sampling1'.format(prefix))
        output_soft_mask = UpSampling2D(size= [1, 2], name= '{}_soft_mask_branch_up_sampling1'.format(prefix))(output_soft_mask)

        #  output
        output_soft_mask = Conv2D(filters= filters, kernel_size= 1, name= '{}_soft_mask_branch_output_1'.format(prefix))(output_soft_mask)
        output_soft_mask = Conv2D(filters= filters, kernel_size= 1, name= '{}_soft_mask_branch_output_2'.format(prefix))(output_soft_mask)
        output_soft_mask = Activation('sigmoid', name= '{}_soft_mask_branch_output_act'.format(prefix))(output_soft_mask)

        # attention: add the soft mask with residual mode
        output_attention = Add(name= '{}_soft_mask_residual_apply'.format(prefix))([output_trunk, Multiply(name= '{}_soft_mask_apply'.format(prefix))([output_soft_mask, output_trunk])])

        for i in range(p):
            output_attention = self.__residual_block(output_attention, filters= filters, scope = '{}_output_{}'.format(prefix, i))

        return output_attention

    def network(self):
        ''''''
        input_layer = Input(shape= self.shape)

        filters = [32, 64, 128, 256, 256, 256, 256]
        # block 0: first feature map
        x = Conv2D(filters= filters[0], kernel_size= 5, strides= 1, padding= 'same', name= 'block_0_conv')(input_layer)
        x = MaxPooling2D(pool_size= [1, 5], strides= [1, 1], padding= 'same', name= 'block_0_pooling')(input_layer)
        x = self.__attention_block(x, filters= filters[0], scope= 'block_0')

        # block 1: second feature map
        x = self.__residual_block(x, filters= filters[1], scope= 'block_1')
        x = MaxPooling2D(pool_size= [1, 3], strides= [1, 1], padding= 'same', name= 'block_1_pooling')(x)
        x = self.__attention_block(x, filters= filters[1], scope= 'block_1')

        # block 2: third feature map
        x = self.__residual_block(x, filters= filters[2], scope= 'block_2')
        x = MaxPooling2D(pool_size= [1, 3], strides= [1, 2], padding= 'same', name= 'block_2_pooling')(x)
        x = self.__attention_block(x, filters= filters[2], scope= 'block_2')

        # block 3: fourth feature map
        x = self.__residual_block(x, filters= filters[3], scope= 'block_3')
        x = MaxPooling2D(pool_size= [1, 3], strides= [1, 2], padding= 'same', name= 'block_3_pooling')(x)

        # block 4: fourth feature map
        x = self.__residual_block(x, filters = filters[4], scope= 'block_4')

        # block 5: fourth feature map
        x = self.__residual_block(x, filters = filters[5], scope= 'block_5')

        # block 6: fourth feature map
        x = self.__residual_block(x, filters = filters[6], scope= 'block_6')

        #
        x = AveragePooling2D(pool_size= [1, 8], strides= [1, 1], padding= 'valid')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        output_layer = Dense(1, activation= 'sigmoid')(x)

        network = Model(inputs= input_layer, outputs= output_layer)

        return network

class ImageDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, image_files, batch_size= 32, image_size= [128, 128], n_classes= 3, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.image_files = image_files
        self.image_size = image_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, batch_number):
        'Generate one batch of data'
        # Generate indexes of the batch
        index_batch = self.indexes[batch_number*self.batch_size : (batch_number + 1) * self.batch_size]

        # Find list of IDs
        image_file_batch = [self.image_files[k] for k in index_batch]

        # Generate data
        X, y = self.__data_generation(image_file_batch)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_file_batch):
        # load
        image_data_list = [np.array(load_img(image_file_batch[i], grayscale= False, target_size= self.image_size))/255 for i in range(len(image_file_batch))]
        # label
        image_label_list = [config.level_label_dict[config.level_zn_en[image_file_batch[i].split('/')[-2]]] for i in range(len(image_file_batch))]

        return np.array(image_data_list), np.array(image_label_list)

if __name__ == '__main__':
    ''''''
    strategy = 'att_resnet'

    print('\n')
    # step 1: get image files
    with utils.timer('scan image files'):
        #image_dir = '{}/raw/色情图片已标记'.format(config.DataBaseDir)
        image_dir = '{}/raw/updated_1109'.format(config.DataBaseDir)
        jpg_image_files = glob.glob('{}/*/*.jpg'.format(image_dir))
        png_image_files = glob.glob('{}/*/*.png'.format(image_dir))
        image_files = jpg_image_files + png_image_files
        print('total image files {}'.format(len(image_files)))

    print('\n')
    # step 2: train/valid split
    with utils.timer('split'):
        shuffle(image_files)
        if((config.debug == True) & (config.sampling_ratio < 1.0)):
            image_files = image_files[:int(config.sampling_ratio * len(image_files))]
            print('sampled {:.1f} percentage of dataset'.format(config.sampling_ratio))
        total = len(image_files)
        train_image_files = image_files[:int(0.8 * total)]
        valid_image_files = image_files[int(0.8 * total):]
        print('train/valid = {}/{}'.format(len(train_image_files), len(valid_image_files)))

    # step 3: generator
    with utils.timer('generator'):
        train_generator = ImageDataGenerator(image_files= train_image_files, batch_size= config.batch_size, image_size= config.input_shape[:-1])
        valid_generator = ImageDataGenerator(image_files= valid_image_files, batch_size= config.batch_size, image_size= config.input_shape[:-1])

    # step 4: fitting
    with utils.timer('fitting'):

        # model checkpoint file
        model_weight_dir = '{}/{}/weight'.format(config.ModelRootDir, strategy)
        if(os.path.exists(model_weight_dir) == False):
            os.makedirs(model_weight_dir)
        model_weight_file = '{}/{}.ckpt'.format(model_weight_dir, strategy)

        # callbacks
        monitor_metric, monitor_mode = 'val_loss', 'min'
        model_checkpoint = ModelCheckpoint(model_weight_file, monitor= monitor_metric, mode= monitor_mode, save_best_only=True,verbose=1)
        early_stopping = EarlyStopping(monitor= monitor_metric, mode= monitor_mode, patience=12, verbose=1)
        lr_schedule = ReduceLROnPlateau(monitor= monitor_metric, mode= monitor_mode, factor= 0.5, patience= 6, min_lr= 0.00001,verbose=1)
        callbacks = [model_checkpoint, early_stopping, lr_schedule]

        # optimizer
        opti = Adam(lr= config.learning_rate)

        #  monitors
        precision = keras_metrics.precision(label=1)  # as_keras_metric(tf.metrics.precision)
        recall = keras_metrics.recall(label=1)  # as_keras_metric(tf.metrics.recall)

        net = AttentionResidualNetwork(shape= config.input_shape).network()
        net.compile(loss= 'binary_crossentropy', optimizer= opti, metrics= ['accuracy', precision, recall])

        net.fit_generator(generator= train_generator,
                            validation_data= valid_generator,
                            use_multiprocessing= True,
                            workers= 8, verbose= 2, epochs= config.epochs,
                            callbacks= [model_checkpoint, early_stopping, lr_schedule])
