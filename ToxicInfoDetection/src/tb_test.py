# Created by yuanpingzhou at 11/28/18

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, BatchNormalization, Flatten, Input
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard

import numpy as np

batch_size = 8
C, H, W = 1, 128, 128

input = np.random.uniform(0, 1, (batch_size, H, W, C)).astype(np.float32)
truth = np.random.choice(2, (batch_size,)).astype(np.float32)

input_layer = Input(shape= [128, 128, 1])
x = Conv2D(36, kernel_size= (3, 3), padding= 'same')(input_layer)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(8, kernel_size= (3, 3), padding= 'same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Flatten()(x)
output_layer = Dense(2, activation= 'softmax')(x)
network = Model(input= input_layer, output= output_layer)

network.summary()

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
opti = Adam(lr = 0.0001)
network.compile(optimizer= opti, loss='sparse_categorical_crossentropy')

network.fit(input, truth, epochs= 20, verbose=1, callbacks= [tbCallBack])
