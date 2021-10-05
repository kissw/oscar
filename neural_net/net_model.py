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
from keras.layers import Lambda, Dropout, Flatten, Dense, Activation, concatenate, Concatenate
from keras.layers import Conv2D, Convolution2D, MaxPooling2D, BatchNormalization, Input
from keras import losses, optimizers
import keras.backend as K
import tensorflow as tf

import os
import const
from config import Config

config = Config.neural_net
config_rn = Config.run_neural
def model_pilot_delta():
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import TimeDistributed
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    vel_shape = (None, 1,)
    ######str model#######
    img_str = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_str)
    conv_1 = TimeDistributed(Conv2D(24, (5, 5), strides=(2,2)))(lamb)
    conv_2 = TimeDistributed(Conv2D(36, (5, 5), strides=(2,2)))(conv_1)
    conv_3 = TimeDistributed(Conv2D(48, (5, 5), strides=(2,2)))(conv_2)
    conv_4 = TimeDistributed(Conv2D(64, (3, 3)))(conv_3)
    conv_5 = TimeDistributed(Conv2D(64, (3, 3)), name='conv2d_last')(conv_4)
    flat = TimeDistributed(Flatten())(conv_5)
    
    vel = Input(shape=vel_shape)
    fc_vel = Dense(50, name='fc_vel')(vel)
    concat = Concatenate()([flat, fc_vel])
    
    lstm = LSTM(100, return_sequences=False, dropout=0.2, name='lstm')(concat)
    fc_1 = Dense(100, name='fc_1')(lstm)
    fc_2 = Dense(50, name='fc_2')(fc_1)
    fc_str = Dense(1, name='fc_str')(fc_2)
    fc_t = Dense(1, name='fc_t')(fc_2)
    fc_b = Dense(1, name='fc_b')(fc_2)
    
    model = Model(inputs=[img_str, vel], outputs=[fc_str, fc_t, fc_b])

    return model
    
def model_pilot_vel():
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    vel_shape = (1,)
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (5, 5), strides=(2,2))(lamb)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2))(conv_1)
    conv_3 = Conv2D(48, (5, 5), strides=(2,2))(conv_2)
    conv_4 = Conv2D(64, (3, 3))(conv_3)
    conv_5 = Conv2D(64, (3, 3), name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(100, name='fc_1')(flat)
    fc_2 = Dense(50, name='fc_2')(fc_1)
    
    ######vel model#######
    vel_input = Input(shape=vel_shape)
    
    ######concat##########
    concat_img_vel = concatenate([fc_2, vel_input])
    fc_3 = Dense(10, name='fc_3')(concat_img_vel)
    fc_str = Dense(1, name='fc_str')(fc_3)
    fc_t = Dense(1, name='fc_t')(fc_3)
    fc_b = Dense(1, name='fc_b')(fc_3)
    
    model = Model(inputs=[img_input, vel_input], outputs=[fc_str, fc_t, fc_b])

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
        if config['network_type'] == const.NET_TYPE_PILOT_VEL:
            self.model = model_pilot_vel()
        elif config['network_type'] == const.NET_TYPE_PILOT_DELTA:
            self.model = model_pilot_delta()
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
        # if config['steering_angle_tolerance'] == 0.0:
        #     self.model.compile(loss=losses.mean_squared_error,
        #               optimizer=optimizers.Adam(),
        #               metrics=['accuracy'])
        # else:
        #     self.model.compile(loss=losses.mean_squared_error,
        #               optimizer=optimizers.Adam(),
        #               metrics=['accuracy', self._mean_squared_error])


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

