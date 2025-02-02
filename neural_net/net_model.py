#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
@author: ninad#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

from keras.models import Sequential, Model
from keras.layers import Lambda, Dropout, Flatten, Dense, Activation, Concatenate
from keras.layers import Conv2D, Convolution2D, BatchNormalization, Input
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, Add
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras import losses, optimizers
import keras.backend as K
import tensorflow as tf

import os
import const
from config import Config

config = Config.neural_net
config_rn = Config.run_neural

#model_XXX_s : input image square
#model_XXX_r : input image rectangular

def model_pilotnet_s():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])

    ######img model#######
    img_input = Input(shape=input_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (5, 5), strides=(2,2), activation='relu', name='conv2d_1')(lamb)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2), activation='relu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(48, (5, 5), strides=(2,2), activation='relu', name='conv2d_3')(conv_2)
    conv_4 = Conv2D(64, (3, 3), activation='relu', name='conv2d_4')(conv_3)
    conv_5 = Conv2D(64, (3, 3), activation='relu', name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(100, activation='relu', name='fc_1')(flat)
    fc_2 = Dense(50 , activation='relu', name='fc_2')(fc_1)
    fc_3 = Dense(10 , activation='relu', name='fc_3')(fc_2)
    fc_last = Dense(config['num_outputs'], name='fc_str')(fc_3)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model

def model_pilotnet_r():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])

    ######img model#######
    img_input = Input(shape=input_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (5, 5), strides=(2,2), activation='relu', name='conv2d_1')(lamb)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2), activation='relu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(48, (5, 5), strides=(2,2), activation='relu', name='conv2d_3')(conv_2)
    conv_4 = Conv2D(64, (3, 3), activation='relu', name='conv2d_4')(conv_3)
    conv_5 = Conv2D(64, (3, 3), activation='relu', name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(100, activation='relu', name='fc_1')(flat)
    fc_2 = Dense(50 , activation='relu', name='fc_2')(fc_1)
    fc_3 = Dense(10 , activation='relu', name='fc_3')(fc_2)
    fc_last = Dense(config['num_outputs'], name='fc_str')(fc_3)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model


def model_pilotnet_s_7():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])

    ######img model#######
    img_input = Input(shape=input_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (7, 7), strides=(2,2), activation='relu', name='conv2d_1')(lamb)
    conv_2 = Conv2D(36, (7, 7), strides=(2,2), activation='relu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(48, (7, 7), strides=(2,2), activation='relu', name='conv2d_3')(conv_2)
    conv_4 = Conv2D(64, (5, 5), activation='relu', name='conv2d_4')(conv_3)
    conv_5 = Conv2D(64, (5, 5), activation='relu', name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(100, activation='relu', name='fc_1')(flat)
    fc_2 = Dense(50 , activation='relu', name='fc_2')(fc_1)
    fc_3 = Dense(10 , activation='relu', name='fc_3')(fc_2)
    fc_last = Dense(config['num_outputs'], name='fc_str')(fc_3)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model

def model_pilotnet_s_3():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])

    ######img model#######
    img_input = Input(shape=input_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (3, 3), strides=(2,2), activation='relu', name='conv2d_1')(lamb)
    conv_2 = Conv2D(36, (3, 3), strides=(2,2), activation='relu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(48, (3, 3), strides=(2,2), activation='relu', name='conv2d_3')(conv_2)
    conv_4 = Conv2D(64, (3, 3), activation='relu', name='conv2d_4')(conv_3)
    conv_5 = Conv2D(64, (3, 3), activation='relu', name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(100, activation='relu', name='fc_1')(flat)
    fc_2 = Dense(50 , activation='relu', name='fc_2')(fc_1)
    fc_3 = Dense(10 , activation='relu', name='fc_3')(fc_2)
    fc_last = Dense(config['num_outputs'], name='fc_str')(fc_3)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model

def model_pilotnet_s_9():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])

    ######img model#######
    img_input = Input(shape=input_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (9, 9), strides=(2,2), activation='relu', name='conv2d_1')(lamb)
    conv_2 = Conv2D(36, (9, 9), strides=(2,2), activation='relu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(48, (9, 9), strides=(2,2), activation='relu', name='conv2d_3')(conv_2)
    conv_4 = Conv2D(64, (7, 7), activation='relu', name='conv2d_4')(conv_3)
    conv_5 = Conv2D(64, (7, 7), activation='relu', name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(100, activation='relu', name='fc_1')(flat)
    fc_2 = Dense(50 , activation='relu', name='fc_2')(fc_1)
    fc_3 = Dense(10 , activation='relu', name='fc_3')(fc_2)
    fc_last = Dense(config['num_outputs'], name='fc_str')(fc_3)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model

def model_bimi_s():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])

    ######img model#######
    img_input = Input(shape=input_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (5, 5), strides=(2,2), activation='relu', name='conv2d_1')(lamb)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2), activation='relu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(48, (5, 5), strides=(2,2), activation='relu', name='conv2d_3')(conv_2)
    conv_4 = Conv2D(64, (3, 3), activation='relu', name='conv2d_4')(conv_3)
    conv_5 = Conv2D(64, (3, 3), activation='relu', name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(1000, activation='relu', name='fc_1')(flat)
    fc_2 = Dense(100 , activation='relu', name='fc_2')(fc_1)
    fc_3 = Dense(50 , activation='relu', name='fc_3')(fc_2)
    fc_4 = Dense(10 , activation='relu', name='fc_4')(fc_3)
    fc_last = Dense(config['num_outputs'], name='fc_str')(fc_4)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model

def model_bimi_r():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])

    ######img model#######
    img_input = Input(shape=input_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (5, 5), strides=(2,2), activation='relu', name='conv2d_1')(lamb)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2), activation='relu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(48, (5, 5), strides=(2,2), activation='relu', name='conv2d_3')(conv_2)
    conv_4 = Conv2D(64, (3, 3), activation='relu', name='conv2d_4')(conv_3)
    conv_5 = Conv2D(64, (3, 3), activation='relu', name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(1000, activation='relu', name='fc_1')(flat)
    fc_2 = Dense(100 , activation='relu', name='fc_2')(fc_1)
    fc_3 = Dense(50 , activation='relu', name='fc_3')(fc_2)
    fc_4 = Dense(10 , activation='relu', name='fc_4')(fc_3)
    fc_last = Dense(config['num_outputs'], name='fc_str')(fc_4)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model

def model_conjoin_r(): #
    from keras.layers import add, Concatenate, ELU, UpSampling2D
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1  = Conv2D(64, (8, 8), strides=(2,2), activation='relu', name='conv_1')(lamb)
    conv_2  = Conv2D(64, (6, 6), strides=(2,2), activation='relu', name='conv_2')(conv_1)
    conv_3_1= Conv2D(128, (5, 5), strides=(2,2), padding='same', activation='relu', name='conv_3_1')(conv_2)
    conv_3_2= Conv2D(128, (3, 3), strides=(2,2), padding='same', activation='relu', name='conv_3_2')(conv_2)
    conc_1  = Concatenate(axis=3)([conv_3_1, conv_3_2])
    
    conv_4_1= Conv2D(256, (3, 3), activation='relu')(conc_1)
    conv_4_2= Conv2D(256, (3, 3), activation='relu')(conv_3_2)
    conc_2  = Concatenate(axis=3)([conv_4_1, conv_4_2])
    
    conv_5_1= Conv2D(512, (3, 3), activation='relu', name='conv_5_1')(conc_2)
    conv_5_2= Conv2D(512, (3, 3), activation='relu', name='conv_5_2')(conv_4_2)
    conc_3  = Concatenate(axis=3)([conv_5_1, conv_5_2])
    conv_6  = Conv2D(512, (3, 3), activation='relu', name='conv2d_last')(conc_3)
    
    flat_1  = Flatten()(conv_6)
    fc_1 = Dense(1000, activation='relu', name='fc_1')(flat_1)
    fc_2 = Dense(100,  activation='relu', name='fc_2')(fc_1)
    fc_3 = Dense(50,   activation='relu', name='fc_3')(fc_2)
    fc_last = Dense(1, name='fc_str')(fc_3)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model

def model_conjoin_s(): #
    from keras.layers import add, Concatenate, ELU, UpSampling2D
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1  = Conv2D(64, (8, 8), strides=(2,2), activation='relu', name='conv_1')(lamb)
    conv_2  = Conv2D(64, (6, 6), strides=(2,2), activation='relu', name='conv_2')(conv_1)
    conv_3_1= Conv2D(128, (5, 5), strides=(2,2), padding='same', activation='relu', name='conv_3_1')(conv_2)
    conv_3_2= Conv2D(128, (3, 3), strides=(2,2), padding='same', activation='relu', name='conv_3_2')(conv_2)
    conc_1  = Concatenate(axis=3)([conv_3_1, conv_3_2])
    
    conv_4_1= Conv2D(256, (3, 3), activation='relu')(conc_1)
    conv_4_2= Conv2D(256, (3, 3), activation='relu')(conv_3_2)
    conc_2  = Concatenate(axis=3)([conv_4_1, conv_4_2])
    
    conv_5_1= Conv2D(512, (3, 3), activation='relu', name='conv_5_1')(conc_2)
    conv_5_2= Conv2D(512, (3, 3), activation='relu', name='conv_5_2')(conv_4_2)
    conc_3  = Concatenate(axis=3)([conv_5_1, conv_5_2])
    conv_6  = Conv2D(512, (3, 3), activation='relu', name='conv2d_last')(conc_3)
    
    flat_1  = Flatten()(conv_6)
    fc_1 = Dense(1000, activation='relu', name='fc_1')(flat_1)
    fc_2 = Dense(100,  activation='relu', name='fc_2')(fc_1)
    fc_3 = Dense(50,   activation='relu', name='fc_3')(fc_2)
    fc_last = Dense(1, name='fc_str')(fc_3)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model

def model_vs_r():
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(3, (1, 1), activation='relu')(lamb)
    conv_2 = Conv2D(32, (3, 3), activation='relu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(32, (3, 3), activation='relu', name='conv2d_3')(conv_2)
    pool_1 = MaxPooling2D(pool_size=(2, 2), name='maxpool_1')(conv_3)
    conv_4 = Conv2D(64, (3, 3), activation='relu', name='conv2d_5')(pool_1)
    conv_5 = Conv2D(64, (3, 3), activation='relu', name='conv2d_6')(conv_4)
    pool_2 = MaxPooling2D(pool_size=(2, 2), name='maxpool_2')(conv_5)
    conv_6 = Conv2D(128, (3, 3), activation='relu', name='conv2d_8')(pool_2)
    conv_7 = Conv2D(128, (3, 3), activation='relu', name='conv2d_last')(conv_6)
    pool_3 = MaxPooling2D(pool_size=(2, 2), name='maxpool_3')(conv_7)
    flat_1 = Flatten()(pool_3)
    fc_1 = Dense(512, activation='relu', name='fc_1')(flat_1)
    fc_2 = Dense(64,  activation='relu', name='fc_2')(fc_1)
    fc_3 = Dense(16,  activation='relu', name='fc_3')(fc_2)
    fc_last = Dense(1, name='fc_str')(fc_3)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model

def model_vs_s():
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(3, (1, 1), activation='relu')(lamb)
    conv_2 = Conv2D(32, (3, 3), activation='relu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(32, (3, 3), activation='relu', name='conv2d_3')(conv_2)
    pool_1 = MaxPooling2D(pool_size=(2, 2), name='maxpool_1')(conv_3)
    conv_4 = Conv2D(64, (3, 3), activation='relu', name='conv2d_5')(pool_1)
    conv_5 = Conv2D(64, (3, 3), activation='relu', name='conv2d_6')(conv_4)
    pool_2 = MaxPooling2D(pool_size=(2, 2), name='maxpool_2')(conv_5)
    conv_6 = Conv2D(128, (3, 3), activation='relu', name='conv2d_8')(pool_2)
    conv_7 = Conv2D(128, (3, 3), activation='relu', name='conv2d_last')(conv_6)
    pool_3 = MaxPooling2D(pool_size=(2, 2), name='maxpool_3')(conv_7)
    flat_1 = Flatten()(pool_3)
    fc_1 = Dense(512, activation='relu', name='fc_1')(flat_1)
    fc_2 = Dense(64,  activation='relu', name='fc_2')(fc_1)
    fc_3 = Dense(16,  activation='relu', name='fc_3')(fc_2)
    fc_last = Dense(1, name='fc_str')(fc_3)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model


def model_dave2sky_r():
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (5, 5), strides=(2,2), activation='relu')(lamb)
    conv_1_bn = BatchNormalization()(conv_1)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2), activation='relu')(conv_1_bn)
    conv_2_bn = BatchNormalization()(conv_2)
    conv_3 = Conv2D(48, (5, 5), strides=(2,2), activation='relu')(conv_2_bn)
    conv_3_bn = BatchNormalization()(conv_3)
    conv_4 = Conv2D(64, (3, 3), activation='relu')(conv_3_bn)
    conv_5 = Conv2D(64, (3, 3), name='conv2d_last', activation='relu')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(32, name='fc_1')(flat)
    fc_last = Dense(config['num_outputs'], name='fc_str')(fc_1)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model

def model_dave2sky_s():
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (5, 5), strides=(2,2), activation='relu')(lamb)
    conv_1_bn = BatchNormalization()(conv_1)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2), activation='relu')(conv_1_bn)
    conv_2_bn = BatchNormalization()(conv_2)
    conv_3 = Conv2D(48, (5, 5), strides=(2,2), activation='relu')(conv_2_bn)
    conv_3_bn = BatchNormalization()(conv_3)
    conv_4 = Conv2D(64, (3, 3), activation='relu')(conv_3_bn)
    conv_5 = Conv2D(64, (3, 3), name='conv2d_last', activation='relu')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(32, name='fc_1')(flat)
    fc_last = Dense(config['num_outputs'], name='fc_str')(fc_1)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model

def model_vgg16_r():    
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
        
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(64, (3, 3), padding="same", activation='relu', name='conv_1')(lamb)
    conv_2 = Conv2D(64, (3, 3), padding="same", activation='relu', name='conv_2')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool_1')(conv_2)
    conv_3 = Conv2D(128, (3, 3), padding="same", activation='relu', name='conv_3')(pool_1)
    conv_4 = Conv2D(128, (3, 3), padding="same", activation='relu', name='conv_4')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool_2')(conv_4)
    conv_5 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_5')(pool_2)
    conv_6 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_6')(conv_5)
    conv_7 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_7')(conv_6)
    pool_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool_3')(conv_7)
    conv_8 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_8')(pool_3)
    conv_9 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_9')(conv_8)
    conv_10= Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_10')(conv_9)
    pool_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool_4')(conv_10)
    conv_11= Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_11')(pool_4)
    conv_12= Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_12')(conv_11)
    conv_13= Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_13')(conv_12)
    pool_5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool_5')(conv_13)
    flat = Flatten()(pool_5)
    fc_1 = Dense(4096, activation='relu', name='fc_1')(flat)
    fc_2 = Dense(4096, activation='relu', name='fc_2')(fc_1)
    fc_last = Dense(1, activation='linear', name='fc_str')(fc_2)
    
    model = Model(inputs=img_input, outputs=fc_last)
    
    return model

def model_vgg16_s():    
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
        
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(64, (3, 3), padding="same", activation='relu', name='conv_1')(lamb)
    conv_2 = Conv2D(64, (3, 3), padding="same", activation='relu', name='conv_2')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool_1')(conv_2)
    conv_3 = Conv2D(128, (3, 3), padding="same", activation='relu', name='conv_3')(pool_1)
    conv_4 = Conv2D(128, (3, 3), padding="same", activation='relu', name='conv_4')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool_2')(conv_4)
    conv_5 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_5')(pool_2)
    conv_6 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_6')(conv_5)
    conv_7 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_7')(conv_6)
    pool_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool_3')(conv_7)
    conv_8 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_8')(pool_3)
    conv_9 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_9')(conv_8)
    conv_10= Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_10')(conv_9)
    pool_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool_4')(conv_10)
    conv_11= Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_11')(pool_4)
    conv_12= Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_12')(conv_11)
    conv_13= Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_13')(conv_12)
    pool_5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool_5')(conv_13)
    flat = Flatten()(pool_5)
    fc_1 = Dense(4096, activation='relu', name='fc_1')(flat)
    fc_2 = Dense(4096, activation='relu', name='fc_2')(fc_1)
    fc_last = Dense(1, activation='linear', name='fc_str')(fc_2)
    
    model = Model(inputs=img_input, outputs=fc_last)
    
    return model

def model_alexnet_r(): 
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
        
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(96, (11, 11), strides=(4,4), padding="same", activation='relu', name='conv_1')(lamb)
    conv_1_bn = BatchNormalization()(conv_1)
    conv_1_pl = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool_1')(conv_1_bn)
    
    conv_2 = Conv2D(256, (5, 5), padding="same", activation='relu', name='conv_2')(conv_1_pl)
    conv_2_bn = BatchNormalization()(conv_2)
    conv_2_pl = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool_2')(conv_2_bn)
    
    conv_3 = Conv2D(384, (3, 3), padding="same", activation='relu', name='conv_3')(conv_2_pl)
    conv_4 = Conv2D(384, (3, 3), padding="same", activation='relu', name='conv_4')(conv_3)
    conv_5 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_5')(conv_4)
    conv_5_pl = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool_3')(conv_5)
    
    flat = Flatten()(conv_5_pl)
    fc_1 = Dense(4096, activation='relu', name='fc_1')(flat)
    fc_2 = Dense(4096, activation='relu', name='fc_2')(fc_1)
    fc_last = Dense(config['num_outputs'], activation='linear', name='fc_str')(fc_2)
    
    model = Model(inputs=img_input, outputs=fc_last)
    
    return model


def model_alexnet_s(): 
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
        
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(96, (11, 11), strides=(4,4), padding="same", activation='relu', name='conv_1')(lamb)
    conv_1_bn = BatchNormalization()(conv_1)
    conv_1_pl = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool_1')(conv_1_bn)
    
    conv_2 = Conv2D(256, (5, 5), padding="same", activation='relu', name='conv_2')(conv_1_pl)
    conv_2_bn = BatchNormalization()(conv_2)
    conv_2_pl = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool_2')(conv_2_bn)
    
    conv_3 = Conv2D(384, (3, 3), padding="same", activation='relu', name='conv_3')(conv_2_pl)
    conv_4 = Conv2D(384, (3, 3), padding="same", activation='relu', name='conv_4')(conv_3)
    conv_5 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_5')(conv_4)
    conv_5_pl = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool_3')(conv_5)
    
    flat = Flatten()(conv_5_pl)
    fc_1 = Dense(4096, activation='relu', name='fc_1')(flat)
    fc_2 = Dense(4096, activation='relu', name='fc_2')(fc_1)
    fc_last = Dense(config['num_outputs'], activation='linear', name='fc_str')(fc_2)
    
    model = Model(inputs=img_input, outputs=fc_last)
    
    return model


def model_resnet18_r(): 
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
        
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    #Block A
    conv_1 = Conv2D(64, (7, 7), strides=(2,2), padding="same", activation='relu', name='conv_1')(lamb)
    conv_1_pl = MaxPooling2D(pool_size=(3, 3), padding="same", strides=(2, 2), name='maxpool_1')(conv_1)
    #Block B
    conv_2_1_1 = Conv2D(64, (3, 3), padding="same", activation='relu', name='conv_2_1_1')(conv_1_pl)
    conv_2_1_2 = Conv2D(64, (3, 3), padding="same", activation='relu', name='conv_2_1_2')(conv_2_1_1)
    conc_1 = Add()([conv_1_pl, conv_2_1_2])
    conv_2_2_1 = Conv2D(64, (3, 3), padding="same", activation='relu', name='conv_2_2_1')(conc_1)
    conv_2_2_2 = Conv2D(64, (3, 3), padding="same", activation='relu', name='conv_2_2_2')(conv_2_2_1)
    conc_2 = Add()([conc_1, conv_2_2_2])
    #Block C
    conv_3_1_0 = Conv2D(128, (1, 1), strides=(2,2), padding="same", activation='relu', name='conv_3_1_3')(conc_2)
    conv_3_1_1 = Conv2D(128, (3, 3), padding="same", activation='relu', name='conv_3_1_1')(conv_3_1_0)
    conv_3_1_2 = Conv2D(128, (3, 3), padding="same", activation='relu', name='conv_3_1_2')(conv_3_1_1)
    conc_3 = Add()([conv_3_1_0, conv_3_1_2])
    conv_3_2_1 = Conv2D(128, (3, 3), padding="same", activation='relu', name='conv_3_2_1')(conc_3)
    conv_3_2_2 = Conv2D(128, (3, 3), padding="same", activation='relu', name='conv_3_2_2')(conv_3_2_1)
    conc_4 = Add()([conc_3, conv_3_2_2])
    #Block D
    conv_4_1_0 = Conv2D(256, (1, 1), strides=(2,2), padding="same", activation='relu', name='conv_4_1_0')(conc_4)
    conv_4_1_1 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_4_1_1')(conv_4_1_0)
    conv_4_1_2 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_4_1_2')(conv_4_1_1)
    conc_5 = Add()([conv_4_1_0, conv_4_1_2])
    conv_4_2_1 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_4_2_1')(conc_5)
    conv_4_2_2 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_4_2_2')(conv_4_2_1)
    conc_6 = Add()([conc_5, conv_4_2_2])
    #Block E
    conv_5_1_0 = Conv2D(512, (1, 1), strides=(2,2), padding="same", activation='relu', name='conv_5_1_0')(conc_6)
    conv_5_1_1 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_5_1_1')(conv_5_1_0)
    conv_5_1_2 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_5_1_2')(conv_5_1_1)
    
    conc_7 = Add()([conv_5_1_0, conv_5_1_2])
    
    conv_5_2_1 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_5_2_1')(conc_7)
    conv_5_2_2 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_5_2_2')(conv_5_2_1)
    
    conc_8 = Add()([conc_7, conv_5_2_2])
    
    # print(conc_8.shape)
    conv_5_pl = GlobalAveragePooling2D()(conc_8)
    # print(conv_5_pl.shape)

    # flat = Flatten()(conv_5_pl)
    fc_last = Dense(config['num_outputs'], activation='linear', name='fc_str')(conv_5_pl)
    
    model = Model(inputs=img_input, outputs=fc_last)
    
    return model


def model_resnet18_s(): 
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
        
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    #Block A
    conv_1 = Conv2D(64, (7, 7), strides=(2,2), padding="same", activation='relu', name='conv_1')(lamb)
    conv_1_pl = MaxPooling2D(pool_size=(3, 3), padding="same", strides=(2, 2), name='maxpool_1')(conv_1)
    #Block B
    conv_2_1_1 = Conv2D(64, (3, 3), padding="same", activation='relu', name='conv_2_1_1')(conv_1_pl)
    conv_2_1_2 = Conv2D(64, (3, 3), padding="same", activation='relu', name='conv_2_1_2')(conv_2_1_1)
    conc_1 = Add()([conv_1_pl, conv_2_1_2])
    conv_2_2_1 = Conv2D(64, (3, 3), padding="same", activation='relu', name='conv_2_2_1')(conc_1)
    conv_2_2_2 = Conv2D(64, (3, 3), padding="same", activation='relu', name='conv_2_2_2')(conv_2_2_1)
    conc_2 = Add()([conc_1, conv_2_2_2])
    #Block C
    conv_3_1_0 = Conv2D(128, (1, 1), strides=(2,2), padding="same", activation='relu', name='conv_3_1_3')(conc_2)
    conv_3_1_1 = Conv2D(128, (3, 3), padding="same", activation='relu', name='conv_3_1_1')(conv_3_1_0)
    conv_3_1_2 = Conv2D(128, (3, 3), padding="same", activation='relu', name='conv_3_1_2')(conv_3_1_1)
    conc_3 = Add()([conv_3_1_0, conv_3_1_2])
    conv_3_2_1 = Conv2D(128, (3, 3), padding="same", activation='relu', name='conv_3_2_1')(conc_3)
    conv_3_2_2 = Conv2D(128, (3, 3), padding="same", activation='relu', name='conv_3_2_2')(conv_3_2_1)
    conc_4 = Add()([conc_3, conv_3_2_2])
    #Block D
    conv_4_1_0 = Conv2D(256, (1, 1), strides=(2,2), padding="same", activation='relu', name='conv_4_1_0')(conc_4)
    conv_4_1_1 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_4_1_1')(conv_4_1_0)
    conv_4_1_2 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_4_1_2')(conv_4_1_1)
    conc_5 = Add()([conv_4_1_0, conv_4_1_2])
    conv_4_2_1 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_4_2_1')(conc_5)
    conv_4_2_2 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_4_2_2')(conv_4_2_1)
    conc_6 = Add()([conc_5, conv_4_2_2])
    #Block E
    conv_5_1_0 = Conv2D(512, (1, 1), strides=(2,2), padding="same", activation='relu', name='conv_5_1_0')(conc_6)
    conv_5_1_1 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_5_1_1')(conv_5_1_0)
    conv_5_1_2 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_5_1_2')(conv_5_1_1)
    
    conc_7 = Add()([conv_5_1_0, conv_5_1_2])
    
    conv_5_2_1 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_5_2_1')(conc_7)
    conv_5_2_2 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_5_2_2')(conv_5_2_1)
    
    conc_8 = Add()([conc_7, conv_5_2_2])
    
    # print(conc_8.shape)
    conv_5_pl = GlobalAveragePooling2D()(conc_8)
    # print(conv_5_pl.shape)

    # flat = Flatten()(conv_5_pl)
    fc_last = Dense(config['num_outputs'], activation='linear', name='fc_str')(conv_5_pl)
    
    model = Model(inputs=img_input, outputs=fc_last)
    
    return model

def model_pilotnet_lstm_r():

    # redefine input_shape to add one more dims
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    
    input_img = Input(shape=img_shape, name='input_image')
    lamb      = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0), name='lamb_img')(input_img)
    conv_1    = TimeDistributed(Convolution2D(24, (5, 5), activation='relu', strides=(2,2)), name='conv_1')(lamb)
    conv_2    = TimeDistributed(Convolution2D(36, (5, 5), activation='relu', strides=(2,2)), name='conv_2')(conv_1)
    conv_3    = TimeDistributed(Convolution2D(48, (5, 5), activation='relu', strides=(2,2)), name='conv_3')(conv_2)
    conv_4    = TimeDistributed(Convolution2D(64, (3, 3), activation='relu'), name='conv_4')(conv_3)
    conv_5    = TimeDistributed(Convolution2D(64, (3, 3), activation='relu'), name='conv2d_last')(conv_4)
    flat      = TimeDistributed(Flatten(), name='flat')(conv_5)
    lstm      = LSTM( 100, return_sequences=False, name='lstm')(flat)
    fc_1      = Dense(100, activation='relu', name='fc_1')(lstm)
    fc_2      = Dense( 50, activation='relu', name='fc_2')(fc_1)
    fc_3      = Dense( 10, activation='relu', name='fc_3')(fc_2)
    fc_last   = Dense(config['num_outputs'], activation='linear', name='fc_last')(fc_3)

    model = Model(inputs=input_img, outputs=fc_last)
    
    return model

def model_pilotnet_lstm_s():

    # redefine input_shape to add one more dims
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    
    input_img = Input(shape=img_shape, name='input_image')
    lamb      = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0), name='lamb_img')(input_img)
    conv_1    = TimeDistributed(Convolution2D(24, (5, 5), activation='relu', strides=(2,2)), name='conv_1')(lamb)
    conv_2    = TimeDistributed(Convolution2D(36, (5, 5), activation='relu', strides=(2,2)), name='conv_2')(conv_1)
    conv_3    = TimeDistributed(Convolution2D(48, (5, 5), activation='relu', strides=(2,2)), name='conv_3')(conv_2)
    conv_4    = TimeDistributed(Convolution2D(64, (3, 3), activation='relu'), name='conv_4')(conv_3)
    conv_5    = TimeDistributed(Convolution2D(64, (3, 3), activation='relu'), name='conv2d_last')(conv_4)
    flat      = TimeDistributed(Flatten(), name='flat')(conv_5)
    lstm      = LSTM( 100, return_sequences=False, name='lstm')(flat)
    fc_1      = Dense(100, activation='relu', name='fc_1')(lstm)
    fc_2      = Dense( 50, activation='relu', name='fc_2')(fc_1)
    fc_3      = Dense( 10, activation='relu', name='fc_3')(fc_2)
    fc_last   = Dense(config['num_outputs'], activation='linear', name='fc_last')(fc_3)

    model = Model(inputs=input_img, outputs=fc_last)
    
    return model


def model_bimi_lstm_s():
    input_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])

    ######img model#######
    img_input = Input(shape=input_shape)
    lamb   = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0))(img_input)
    conv_1 = TimeDistributed(Conv2D(24, (5, 5), strides=(2,2), activation='relu'), name='conv2d_1')(lamb)
    conv_2 = TimeDistributed(Conv2D(36, (5, 5), strides=(2,2), activation='relu'), name='conv2d_2')(conv_1)
    conv_3 = TimeDistributed(Conv2D(48, (5, 5), strides=(2,2), activation='relu'), name='conv2d_3')(conv_2)
    conv_4 = TimeDistributed(Conv2D(64, (3, 3), activation='relu'), name='conv2d_4')(conv_3)
    conv_5 = TimeDistributed(Conv2D(64, (3, 3), activation='relu'), name='conv2d_last')(conv_4)
    flat   = TimeDistributed(Flatten(), name='flat')(conv_5)
    lstm   = LSTM( 100, return_sequences=False, name='lstm')(flat)
    fc_1   = Dense(1000, activation='relu', name='fc_1')(lstm)
    fc_2   = Dense(100 , activation='relu', name='fc_2')(fc_1)
    fc_3   = Dense(50 , activation='relu', name='fc_3')(fc_2)
    fc_4   = Dense(10 , activation='relu', name='fc_4')(fc_3)
    fc_last = Dense(config['num_outputs'], name='fc_str')(fc_4)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model

def model_bimi_lstm_r():
    input_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])

    ######img model#######
    img_input = Input(shape=input_shape)
    lamb   = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0))(img_input)
    conv_1 = TimeDistributed(Conv2D(24, (5, 5), strides=(2,2), activation='relu'), name='conv2d_1')(lamb)
    conv_2 = TimeDistributed(Conv2D(36, (5, 5), strides=(2,2), activation='relu'), name='conv2d_2')(conv_1)
    conv_3 = TimeDistributed(Conv2D(48, (5, 5), strides=(2,2), activation='relu'), name='conv2d_3')(conv_2)
    conv_4 = TimeDistributed(Conv2D(64, (3, 3), activation='relu'), name='conv2d_4')(conv_3)
    conv_5 = TimeDistributed(Conv2D(64, (3, 3), activation='relu'), name='conv2d_last')(conv_4)
    flat   = TimeDistributed(Flatten(), name='flat')(conv_5)
    lstm   = LSTM( 100, return_sequences=False, name='lstm')(flat)
    fc_1   = Dense(1000, activation='relu', name='fc_1')(lstm)
    fc_2   = Dense(100 , activation='relu', name='fc_2')(fc_1)
    fc_3   = Dense(50 , activation='relu', name='fc_3')(fc_2)
    fc_4   = Dense(10 , activation='relu', name='fc_4')(fc_3)
    fc_last = Dense(config['num_outputs'], name='fc_str')(fc_4)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model


def model_alexnet_lstm_s(): 
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
        
    img_input = Input(shape=img_shape)
    lamb      = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0))(img_input)
    conv_1    = TimeDistributed(Conv2D(96, (11, 11), strides=(4,4), padding="same", activation='relu'), name='conv_1')(lamb)
    conv_1_bn = TimeDistributed(BatchNormalization())(conv_1)
    conv_1_pl = TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)), name='maxpool_1')(conv_1_bn)
    conv_2    = TimeDistributed(Conv2D(256, (5, 5), padding="same", activation='relu'), name='conv_2')(conv_1_pl)
    conv_2_bn = TimeDistributed(BatchNormalization())(conv_2)
    conv_2_pl = TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)), name='maxpool_2')(conv_2_bn)
    conv_3    = TimeDistributed(Conv2D(384, (3, 3), padding="same", activation='relu'), name='conv_3')(conv_2_pl)
    conv_4    = TimeDistributed(Conv2D(384, (3, 3), padding="same", activation='relu'), name='conv_4')(conv_3)
    conv_5    = TimeDistributed(Conv2D(256, (3, 3), padding="same", activation='relu'), name='conv_5')(conv_4)
    conv_5_pl = TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)), name='maxpool_3')(conv_5)
    flat      = TimeDistributed(Flatten())(conv_5_pl)
    lstm   = LSTM( 100, return_sequences=False, name='lstm')(flat)
    fc_1 = Dense(4096, activation='relu', name='fc_1')(lstm)
    fc_2 = Dense(4096, activation='relu', name='fc_2')(fc_1)
    fc_last = Dense(config['num_outputs'], activation='linear', name='fc_str')(fc_2)
    
    model = Model(inputs=img_input, outputs=fc_last)
    
    return model


def model_alexnet_lstm_r(): 
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
        
    img_input = Input(shape=img_shape)
    lamb      = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0))(img_input)
    conv_1    = TimeDistributed(Conv2D(96, (11, 11), strides=(4,4), padding="same", activation='relu'), name='conv_1')(lamb)
    conv_1_bn = TimeDistributed(BatchNormalization())(conv_1)
    conv_1_pl = TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)), name='maxpool_1')(conv_1_bn)
    conv_2    = TimeDistributed(Conv2D(256, (5, 5), padding="same", activation='relu'), name='conv_2')(conv_1_pl)
    conv_2_bn = TimeDistributed(BatchNormalization())(conv_2)
    conv_2_pl = TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)), name='maxpool_2')(conv_2_bn)
    conv_3    = TimeDistributed(Conv2D(384, (3, 3), padding="same", activation='relu'), name='conv_3')(conv_2_pl)
    conv_4    = TimeDistributed(Conv2D(384, (3, 3), padding="same", activation='relu'), name='conv_4')(conv_3)
    conv_5    = TimeDistributed(Conv2D(256, (3, 3), padding="same", activation='relu'), name='conv_5')(conv_4)
    conv_5_pl = TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)), name='maxpool_3')(conv_5)
    flat      = TimeDistributed(Flatten())(conv_5_pl)
    lstm   = LSTM( 100, return_sequences=False, name='lstm')(flat)
    fc_1 = Dense(4096, activation='relu', name='fc_1')(lstm)
    fc_2 = Dense(4096, activation='relu', name='fc_2')(fc_1)
    fc_last = Dense(config['num_outputs'], activation='linear', name='fc_str')(fc_2)
    
    model = Model(inputs=img_input, outputs=fc_last)
    
    return model


def model_resnet18_lstm_s(): 
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
        
    img_input = Input(shape=img_shape)
    lamb        = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0))(img_input)
    conv_1      = TimeDistributed(Conv2D(64, (7, 7), strides=(2,2), padding="same", activation='relu'), name='conv_1')(lamb)
    conv_1_pl   = TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding="same", strides=(2, 2)), name='maxpool_1')(conv_1)
    conv_2_1_1  = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation='relu'), name='conv_2_1_1')(conv_1_pl)
    conv_2_1_2  = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation='relu'), name='conv_2_1_2')(conv_2_1_1)
    conc_1      = Add()([conv_1_pl, conv_2_1_2])
    conv_2_2_1  = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation='relu'), name='conv_2_2_1')(conc_1)
    conv_2_2_2  = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation='relu'), name='conv_2_2_2')(conv_2_2_1)
    conc_2      = Add()([conc_1, conv_2_2_2])
    conv_3_1_0  = TimeDistributed(Conv2D(128, (1, 1), strides=(2,2), padding="same", activation='relu'), name='conv_3_1_3')(conc_2)
    conv_3_1_1  = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation='relu'), name='conv_3_1_1')(conv_3_1_0)
    conv_3_1_2  = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation='relu'), name='conv_3_1_2')(conv_3_1_1)
    conc_3      = Add()([conv_3_1_0, conv_3_1_2])
    conv_3_2_1  = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation='relu'), name='conv_3_2_1')(conc_3)
    conv_3_2_2  = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation='relu'), name='conv_3_2_2')(conv_3_2_1)
    conc_4      = Add()([conc_3, conv_3_2_2])
    conv_4_1_0  = TimeDistributed(Conv2D(256, (1, 1), strides=(2,2), padding="same", activation='relu'), name='conv_4_1_0')(conc_4)
    conv_4_1_1  = TimeDistributed(Conv2D(256, (3, 3), padding="same", activation='relu'), name='conv_4_1_1')(conv_4_1_0)
    conv_4_1_2  = TimeDistributed(Conv2D(256, (3, 3), padding="same", activation='relu'), name='conv_4_1_2')(conv_4_1_1)
    conc_5      = Add()([conv_4_1_0, conv_4_1_2])
    conv_4_2_1  = TimeDistributed(Conv2D(256, (3, 3), padding="same", activation='relu'), name='conv_4_2_1')(conc_5)
    conv_4_2_2  = TimeDistributed(Conv2D(256, (3, 3), padding="same", activation='relu'), name='conv_4_2_2')(conv_4_2_1)
    conc_6      = Add()([conc_5, conv_4_2_2])
    conv_5_1_0  = TimeDistributed(Conv2D(512, (1, 1), strides=(2,2), padding="same", activation='relu'), name='conv_5_1_0')(conc_6)
    conv_5_1_1  = TimeDistributed(Conv2D(512, (3, 3), padding="same", activation='relu'), name='conv_5_1_1')(conv_5_1_0)
    conv_5_1_2  = TimeDistributed(Conv2D(512, (3, 3), padding="same", activation='relu'), name='conv_5_1_2')(conv_5_1_1)
    conc_7      = Add()([conv_5_1_0, conv_5_1_2])
    conv_5_2_1  = TimeDistributed(Conv2D(512, (3, 3), padding="same", activation='relu'), name='conv_5_2_1')(conc_7)
    conv_5_2_2  = TimeDistributed(Conv2D(512, (3, 3), padding="same", activation='relu'), name='conv_5_2_2')(conv_5_2_1)
    conc_8      = Add()([conc_7, conv_5_2_2])
    conv_5_pl   = TimeDistributed(GlobalAveragePooling2D())(conc_8)
    # print(conv_5_pl.shape)
    lstm   = LSTM( 100, return_sequences=False, name='lstm')(conv_5_pl)

    # flat = Flatten()(conv_5_pl)
    fc_last = Dense(config['num_outputs'], activation='linear', name='fc_str')(lstm)
    
    model = Model(inputs=img_input, outputs=fc_last)
    
    return model

def model_resnet18_lstm_r(): 
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
        
    img_input = Input(shape=img_shape)
    lamb        = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0))(img_input)
    conv_1      = TimeDistributed(Conv2D(64, (7, 7), strides=(2,2), padding="same", activation='relu'), name='conv_1')(lamb)
    conv_1_pl   = TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding="same", strides=(2, 2)), name='maxpool_1')(conv_1)
    conv_2_1_1  = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation='relu'), name='conv_2_1_1')(conv_1_pl)
    conv_2_1_2  = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation='relu'), name='conv_2_1_2')(conv_2_1_1)
    conc_1      = Add()([conv_1_pl, conv_2_1_2])
    conv_2_2_1  = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation='relu'), name='conv_2_2_1')(conc_1)
    conv_2_2_2  = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation='relu'), name='conv_2_2_2')(conv_2_2_1)
    conc_2      = Add()([conc_1, conv_2_2_2])
    conv_3_1_0  = TimeDistributed(Conv2D(128, (1, 1), strides=(2,2), padding="same", activation='relu'), name='conv_3_1_3')(conc_2)
    conv_3_1_1  = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation='relu'), name='conv_3_1_1')(conv_3_1_0)
    conv_3_1_2  = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation='relu'), name='conv_3_1_2')(conv_3_1_1)
    conc_3      = Add()([conv_3_1_0, conv_3_1_2])
    conv_3_2_1  = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation='relu'), name='conv_3_2_1')(conc_3)
    conv_3_2_2  = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation='relu'), name='conv_3_2_2')(conv_3_2_1)
    conc_4      = Add()([conc_3, conv_3_2_2])
    conv_4_1_0  = TimeDistributed(Conv2D(256, (1, 1), strides=(2,2), padding="same", activation='relu'), name='conv_4_1_0')(conc_4)
    conv_4_1_1  = TimeDistributed(Conv2D(256, (3, 3), padding="same", activation='relu'), name='conv_4_1_1')(conv_4_1_0)
    conv_4_1_2  = TimeDistributed(Conv2D(256, (3, 3), padding="same", activation='relu'), name='conv_4_1_2')(conv_4_1_1)
    conc_5      = Add()([conv_4_1_0, conv_4_1_2])
    conv_4_2_1  = TimeDistributed(Conv2D(256, (3, 3), padding="same", activation='relu'), name='conv_4_2_1')(conc_5)
    conv_4_2_2  = TimeDistributed(Conv2D(256, (3, 3), padding="same", activation='relu'), name='conv_4_2_2')(conv_4_2_1)
    conc_6      = Add()([conc_5, conv_4_2_2])
    conv_5_1_0  = TimeDistributed(Conv2D(512, (1, 1), strides=(2,2), padding="same", activation='relu'), name='conv_5_1_0')(conc_6)
    conv_5_1_1  = TimeDistributed(Conv2D(512, (3, 3), padding="same", activation='relu'), name='conv_5_1_1')(conv_5_1_0)
    conv_5_1_2  = TimeDistributed(Conv2D(512, (3, 3), padding="same", activation='relu'), name='conv_5_1_2')(conv_5_1_1)
    conc_7      = Add()([conv_5_1_0, conv_5_1_2])
    conv_5_2_1  = TimeDistributed(Conv2D(512, (3, 3), padding="same", activation='relu'), name='conv_5_2_1')(conc_7)
    conv_5_2_2  = TimeDistributed(Conv2D(512, (3, 3), padding="same", activation='relu'), name='conv_5_2_2')(conv_5_2_1)
    conc_8      = Add()([conc_7, conv_5_2_2])
    conv_5_pl   = TimeDistributed(GlobalAveragePooling2D())(conc_8)
    # print(conv_5_pl.shape)
    lstm   = LSTM( 100, return_sequences=False, name='lstm')(conv_5_pl)

    # flat = Flatten()(conv_5_pl)
    fc_last = Dense(config['num_outputs'], activation='linear', name='fc_str')(lstm)
    
    model = Model(inputs=img_input, outputs=fc_last)
    
    return model


def model_conjoin_lstm_s(): #
    from keras.layers import add, Concatenate, ELU, UpSampling2D
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb    = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0))(img_input)
    conv_1  = TimeDistributed(Conv2D(64, (8, 8), strides=(2,2), activation='relu'), name='conv_1')(lamb)
    conv_2  = TimeDistributed(Conv2D(64, (6, 6), strides=(2,2), activation='relu'), name='conv_2')(conv_1)
    conv_3_1= TimeDistributed(Conv2D(128, (5, 5), strides=(2,2), padding='same', activation='relu'), name='conv_3_1')(conv_2)
    conv_3_2= TimeDistributed(Conv2D(128, (3, 3), strides=(2,2), padding='same', activation='relu'), name='conv_3_2')(conv_2)
    conc_1  = Concatenate(axis=3)([conv_3_1, conv_3_2])
    conv_4_1= TimeDistributed(Conv2D(256, (3, 3), activation='relu'))(conc_1)
    conv_4_2= TimeDistributed(Conv2D(256, (3, 3), activation='relu'))(conv_3_2)
    conc_2  = Concatenate(axis=3)([conv_4_1, conv_4_2])
    conv_5_1= TimeDistributed(Conv2D(512, (3, 3), activation='relu'), name='conv_5_1')(conc_2)
    conv_5_2= TimeDistributed(Conv2D(512, (3, 3), activation='relu'), name='conv_5_2')(conv_4_2)
    conc_3  = Concatenate(axis=3)([conv_5_1, conv_5_2])
    conv_6  = TimeDistributed(Conv2D(512, (3, 3), activation='relu'), name='conv2d_last')(conc_3)
    
    flat  = TimeDistributed(Flatten())(conv_6)
    lstm   = LSTM( 100, return_sequences=False, name='lstm')(flat)
    fc_1 = Dense(1000, activation='relu', name='fc_1')(lstm)
    fc_2 = Dense(100,  activation='relu', name='fc_2')(fc_1)
    fc_3 = Dense(50,   activation='relu', name='fc_3')(fc_2)
    fc_last = Dense(1, name='fc_str')(fc_3)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model

def model_conjoin_lstm_r(): #
    from keras.layers import add, Concatenate, ELU, UpSampling2D
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb    = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0))(img_input)
    conv_1  = TimeDistributed(Conv2D(64, (8, 8), strides=(2,2), activation='relu'), name='conv_1')(lamb)
    conv_2  = TimeDistributed(Conv2D(64, (6, 6), strides=(2,2), activation='relu'), name='conv_2')(conv_1)
    conv_3_1= TimeDistributed(Conv2D(128, (5, 5), strides=(2,2), padding='same', activation='relu'), name='conv_3_1')(conv_2)
    conv_3_2= TimeDistributed(Conv2D(128, (3, 3), strides=(2,2), padding='same', activation='relu'), name='conv_3_2')(conv_2)
    conc_1  = Concatenate(axis=3)([conv_3_1, conv_3_2])
    conv_4_1= TimeDistributed(Conv2D(256, (3, 3), activation='relu'))(conc_1)
    conv_4_2= TimeDistributed(Conv2D(256, (3, 3), activation='relu'))(conv_3_2)
    conc_2  = Concatenate(axis=3)([conv_4_1, conv_4_2])
    conv_5_1= TimeDistributed(Conv2D(512, (3, 3), activation='relu'), name='conv_5_1')(conc_2)
    conv_5_2= TimeDistributed(Conv2D(512, (3, 3), activation='relu'), name='conv_5_2')(conv_4_2)
    conc_3  = Concatenate(axis=3)([conv_5_1, conv_5_2])
    conv_6  = TimeDistributed(Conv2D(512, (3, 3), activation='relu'), name='conv2d_last')(conc_3)
    
    flat  = TimeDistributed(Flatten())(conv_6)
    lstm   = LSTM( 100, return_sequences=False, name='lstm')(flat)
    fc_1 = Dense(1000, activation='relu', name='fc_1')(lstm)
    fc_2 = Dense(100,  activation='relu', name='fc_2')(fc_1)
    fc_3 = Dense(50,   activation='relu', name='fc_3')(fc_2)
    fc_last = Dense(1, name='fc_str')(fc_3)
    
    model = Model(inputs=img_input, outputs=fc_last)

    return model

class NetModel:
    def __init__(self, model_path):
        self.model = None
        model_name = model_path[model_path.rfind('/'):] # get folder name
        self.name = model_name.strip('/')

        self.model_path = model_path
        #self.config = Config()

        # to address the error:
        #   Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
        os.environ["CUDA_VISIBLE_DEVICES"]=str(config['gpus'])
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        K.tensorflow_backend.set_session(sess)
        # with tf.device('/cpu:0'):
        self._model()

    ###########################################################################
    #
    def _model(self):
        if config['network_type'] == const.NET_TYPE_PILOT_R:
            self.model = model_pilotnet_r()
        elif config['network_type'] == const.NET_TYPE_PILOT_S:
            self.model = model_pilotnet_s()
        elif config['network_type'] == const.NET_TYPE_PILOT_S_3:
            self.model = model_pilotnet_s()
        elif config['network_type'] == const.NET_TYPE_PILOT_S_7:
            self.model = model_pilotnet_s()
        elif config['network_type'] == const.NET_TYPE_PILOT_S_9:
            self.model = model_pilotnet_s()
        elif config['network_type'] == const.NET_TYPE_BIMI_R:
            self.model = model_bimi_r()
        elif config['network_type'] == const.NET_TYPE_BIMI_S:
            self.model = model_bimi_s()
        elif config['network_type'] == const.NET_TYPE_VS_R:
            self.model = model_vs_r()
        elif config['network_type'] == const.NET_TYPE_VS_S:
            self.model = model_vs_s()
        elif config['network_type'] == const.NET_TYPE_DAVE2SKY_R:
            self.model = model_dave2sky_r()
        elif config['network_type'] == const.NET_TYPE_DAVE2SKY_S:
            self.model = model_dave2sky_s()
        elif config['network_type'] == const.NET_TYPE_VGG16_R:
            self.model = model_vgg16_r()
        elif config['network_type'] == const.NET_TYPE_VGG16_S:
            self.model = model_vgg16_s()
        elif config['network_type'] == const.NET_TYPE_ALEX_R:
            self.model = model_alexnet_r()
        elif config['network_type'] == const.NET_TYPE_ALEX_S:
            self.model = model_alexnet_s()
        elif config['network_type'] == const.NET_TYPE_RES_R:
            self.model = model_resnet18_r()
        elif config['network_type'] == const.NET_TYPE_RES_S:
            self.model = model_resnet18_s()
        elif config['network_type'] == const.NET_TYPE_CONJOIN_R:
            self.model = model_conjoin_r()
        elif config['network_type'] == const.NET_TYPE_CONJOIN_S:
            self.model = model_conjoin_s()
            
            
        elif config['network_type'] == const.NET_TYPE_PILOTwL_R:
            self.model = model_pilotnet_lstm_r()
        elif config['network_type'] == const.NET_TYPE_PILOTwL_S:
            self.model = model_pilotnet_lstm_s()
        elif config['network_type'] == const.NET_TYPE_BIMIwL_R:
            self.model = model_bimi_lstm_r()
        elif config['network_type'] == const.NET_TYPE_BIMIwL_S:
            self.model = model_bimi_lstm_s()
        elif config['network_type'] == const.NET_TYPE_ALEXwL_R:
            self.model = model_alexnet_lstm_r()
        elif config['network_type'] == const.NET_TYPE_ALEXwL_S:
            self.model = model_alexnet_lstm_s()
        elif config['network_type'] == const.NET_TYPE_RESwL_R:
            self.model = model_resnet18_lstm_r()
        elif config['network_type'] == const.NET_TYPE_RESwL_S:
            self.model = model_resnet18_lstm_s()
        elif config['network_type'] == const.NET_TYPE_CONJOINwL_R:
            self.model = model_conjoin_lstm_r()
        elif config['network_type'] == const.NET_TYPE_CONJOINwL_S:
            self.model = model_conjoin_lstm_s()
        else:
            exit('ERROR: Invalid neural network type.')

        self.summary()
        self._compile()



    # ###########################################################################
    # #
    # def _mean_squared_error(self, y_true, y_pred):
    #     diff = K.abs(y_true - y_pred)
    #     if (diff < config['steering_angle_tolerance']) is True:
    #         diff = 0
    #     return K.mean(K.square(diff))

    ###########################################################################
    #
    def _compile(self):
        if config['lstm'] is True:
            learning_rate = config['lstm_lr']
        else:
            learning_rate = config['cnn_lr']
        decay = config['decay']
        self.model.compile(loss=losses.mean_squared_error,
                    optimizer=optimizers.Adam(lr=learning_rate, decay=decay, clipvalue=1), 
                    metrics=['accuracy'])


    ###########################################################################
    #
    # save model
    def save(self, model_name):

        json_string = self.model.to_json()
        #weight_filename = self.model_path + '_' + Config.config_yaml_name \
        #    + '_N' + str(config['network_type'])
        open(model_name+'.json', 'w').write(json_string)
        self.model.save_weights(model_name+'.h5', overwrite=True)


    ###########################################################################
    # model_path = '../data/2007-09-22-12-12-12.
    def weight_load(self, load_model_name):
    
        from keras.models import model_from_json

        json_string = self.model.to_json()
        open(load_model_name+'.json', 'w').write(json_string)
        self.model = model_from_json(open(load_model_name+'.json').read())
        self.model.load_weights(load_model_name)
        self._compile()
    
    
    def load(self):

        from keras.models import model_from_json

        self.model = model_from_json(open(self.model_path+'.json').read())
        self.model.load_weights(self.model_path+'.h5')
        self._compile()

    ###########################################################################
    #
    # show summary
    def summary(self):
        self.model.summary()

