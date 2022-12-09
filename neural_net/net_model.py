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

from keras.models import Sequential, Model, model_from_json
from keras.layers import Lambda, Dropout, Flatten, Dense, Activation, Concatenate, concatenate
from keras.layers import Conv2D, Convolution2D, BatchNormalization, Input, Reshape, Conv2DTranspose
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, Add, UpSampling2D
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

def model_pilotnet():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    input_vel = (1,)
    # input_delta = (3,)
    ######model#######
    img_input = Input(shape=input_shape)
    vel_input = Input(shape=input_vel)
    # delta_input = Input(shape=input_delta)
    
    lamb_str = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    lamb_vel = Lambda(lambda x: x/40)(vel_input)
    # lamb_delta = Lambda(lambda x: x)(delta_input)
    
    conv_1 = Conv2D(24, (5, 5), strides=(2,2), activation='relu', name='conv2d_1')(lamb_str)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2), activation='relu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(64, (5, 5), strides=(2,2), activation='relu', name='conv2d_3')(conv_2)
    conv_4 = Conv2D(64, (3, 3), padding='same',activation='relu', name='conv2d_4')(conv_3)
    conv_5 = Conv2D(64, (3, 3), padding='same',activation='relu', name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    fc_v = Dense(50, activation='relu', name='fc_v')(lamb_vel)
    # fc_c = Dense(50, activation='relu', name='fc_d')(lamb_delta)
    fc_1 = Dense(100, activation='relu', name='fc_1')(flat)
    conc = Concatenate()([fc_1, fc_v])
    # conc = Concatenate()([fc_1, fc_v, fc_c])
    fc_2 = Dense(50 , activation='relu', name='fc_2')(conc)
    fc_3 = Dense(10 , activation='relu', name='fc_3')(fc_2)
    fc_out = Dense(config['num_outputs'], name='fc_out')(fc_3)
    
    model = Model(inputs=[img_input, vel_input], outputs=[fc_out])
    # model = Model(inputs=[img_input, vel_input, delta_input], outputs=[fc_out])
    return model

def pretrained_pilot(base_model_path):
    base_weightsfile = base_model_path+'.h5'
    base_modelfile   = base_model_path+'.json'
    
    base_json_file = open(base_modelfile, 'r')
    base_loaded_model_json = base_json_file.read()
    base_json_file.close()
    base_model = model_from_json(base_loaded_model_json)
    base_model.load_weights(base_weightsfile)
    # if config['style_train'] is True:
    base_model.trainable = False
    
    # print(base_model.get_layer('conv2d_3').output)
    return base_model

def model_style1(base_model_path):

    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    input_vel = (1,)
    input_gvel = (1,)
    ######model#######
    img_input = Input(shape=input_shape)
    vel_input = Input(shape=input_vel)
    gvel_input = Input(shape=input_gvel)
    
    base_model1 = pretrained_pilot(base_model_path)
    base_model2 = pretrained_pilot(base_model_path)
    base_model3 = pretrained_pilot(base_model_path)
    pretrained_model_last = Model(base_model1.input, base_model1.get_layer('fc_out').output, name='base_model_output')
    pretrained_model_conv3 = Model(base_model2.input, base_model2.get_layer('conv2d_3').output, name='base_model_conv2d_3')
    pretrained_model_conv5 = Model(base_model3.input, base_model3.get_layer('conv2d_last').output, name='base_model_conv2d_last')
    # if config['style_train'] is True:
    pretrained_model_conv3.trainable = False
    pretrained_model_conv5.trainable = False
    pretrained_model_last.trainable = False
        
    base_model_last_output = pretrained_model_last([img_input, vel_input])
    base_model_conv3_output = pretrained_model_conv3([img_input, vel_input])
    base_model_conv5_output = pretrained_model_conv5([img_input, vel_input])
    
    add_base_layer = Add()([base_model_conv3_output, base_model_conv5_output])
    fc_vel = Dense(100, activation='relu', name='fc_vel')(vel_input)
    fc_gvel = Dense(100, activation='relu', name='fc_gvel')(gvel_input)
    fc_base_out = Dense(100, activation='relu', name='fc_base_out')(base_model_last_output)
    flat = Flatten()(add_base_layer)
    fc_1 = Dense(500, activation='relu', name='fc_1')(flat)
    conc = Concatenate()([fc_base_out, fc_1, fc_vel, fc_gvel])
    fc_2 = Dense(200, activation='relu', name='fc_2')(conc)
    drop = Dropout(rate=0.2)(fc_2)
    fc_3 = Dense(100, activation='relu', name='fc_3')(drop)
    
    # print(base_model_last_output[0].shape)
    if config['only_thr_brk'] is True: 
        fc_out = Dense(config['num_outputs']-1, name='fc_out')(fc_3)
    else:
        fc_out = Dense(config['num_outputs'], name='fc_out')(fc_3)
    # fc_str = Dense(1, name='fc_str')(base_str)
    # fc_thr = Dense(1, name='fc_thr')(fc_3)
    # fc_brk = Dense(1, name='fc_brk')(fc_3)
    
    model = Model(inputs=[img_input, vel_input, gvel_input], outputs=[fc_out])
    # model = Model(inputs=[img_input, vel_input], outputs=[fc_str, fc_thr, fc_brk])
    return model

def model_style2(base_model_path):

    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    input_vel = (1,)
    input_gvel = (1,)
    ######model#######
    img_input = Input(shape=input_shape)
    vel_input = Input(shape=input_vel)
    gvel_input = Input(shape=input_gvel)
    
    base_model1 = pretrained_pilot(base_model_path)
    base_model2 = pretrained_pilot(base_model_path)
    base_model3 = pretrained_pilot(base_model_path)
    pretrained_model_last = Model(base_model1.input, base_model1.get_layer('fc_out').output, name='base_model_output')
    pretrained_model_conv3 = Model(base_model2.input, base_model2.get_layer('conv2d_3').output, name='base_model_conv2d_3')
    pretrained_model_conv5 = Model(base_model3.input, base_model3.get_layer('conv2d_last').output, name='base_model_conv2d_last')
    # if config['style_train'] is True:
    pretrained_model_conv3.trainable = False
    pretrained_model_conv5.trainable = False
    pretrained_model_last.trainable = False
        
    base_model_last_output = pretrained_model_last([img_input, vel_input])
    base_model_conv3_output = pretrained_model_conv3([img_input, vel_input])
    base_model_conv5_output = pretrained_model_conv5([img_input, vel_input])
    
    add_base_layer = Add()([base_model_conv3_output, base_model_conv5_output])
    fc_vel = Dense(100, activation='relu', name='fc_vel')(vel_input)
    fc_gvel = Dense(100, activation='relu', name='fc_gvel')(gvel_input)
    fc_base_out = Dense(100, activation='relu', name='fc_base_out')(base_model_last_output)
    flat = Flatten()(add_base_layer)
    fc_1 = Dense(500, activation='relu', name='fc_1')(flat)
    conc = Concatenate()([fc_base_out, fc_1, fc_vel, fc_gvel])
    fc_2 = Dense(200, activation='relu', name='fc_2')(conc)
    drop = Dropout(rate=0.2)(fc_2)
    fc_3 = Dense(100, activation='relu', name='fc_3')(drop)
    
    # print(base_model_last_output[0].shape)
    if config['only_thr_brk'] is True: 
        fc_out = Dense(config['num_outputs']-1, name='fc_out')(fc_3)
    else:
        fc_out = Dense(config['num_outputs'], name='fc_out')(fc_3)
    # fc_str = Dense(1, name='fc_str')(base_str)
    # fc_thr = Dense(1, name='fc_thr')(fc_3)
    # fc_brk = Dense(1, name='fc_brk')(fc_3)
    
    model = Model(inputs=[img_input, vel_input, gvel_input], outputs=[fc_out])
    # model = Model(inputs=[img_input, vel_input], outputs=[fc_str, fc_thr, fc_brk])
    return model

def model_nonlstm():
    input_shape = (config['lstm_timestep'], config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    input_vel = (config['lstm_timestep'], 1,)
    ######model#######
    img_input = Input(shape=input_shape)
    vel_input = Input(shape=input_vel)
    lamb_str = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    lamb_vel = Lambda(lambda x: x/40)(vel_input)
    
    conv_1 = TimeDistributed(Conv2D(24, (5, 5), strides=(2,2), activation='relu'), name='conv2d')(lamb_str)
    conv_2 = TimeDistributed(Conv2D(36, (5, 5), strides=(2,2), activation='relu'), name='conv2d_2')(conv_1)
    conv_3 = TimeDistributed(Conv2D(64, (5, 5), strides=(2,2), activation='relu'), name='conv2d_3')(conv_2)
    conv_4 = TimeDistributed(Conv2D(64, (3, 3), padding='same',activation='relu'), name='conv2d_4')(conv_3)
    conv_5 = TimeDistributed(Conv2D(64, (3, 3), padding='same',activation='relu'), name='conv2d_last')(conv_4)
    flat   = TimeDistributed(Flatten())(conv_5)
    fc_v   = TimeDistributed(Dense( 50, activation='relu'), name='fc_v')(lamb_vel)
    fc_1   = TimeDistributed(Dense(100, activation='relu'), name='fc_1')(flat)
    conc   = Concatenate()([fc_1, fc_v])
    fc_2   = TimeDistributed(Dense( 50, activation='relu'), name='fc_2')(conc)
    fc_3   = TimeDistributed(Dense( 10, activation='relu'), name='fc_3')(fc_2)
    # lstm   = LSTM(1, return_sequences=False, name='lstm_c')(fc_3)
    # fc_out = Dense(config['num_outputs'], name='fc_out')(fc_3)
    flat_2 = Flatten()(fc_3)
    fc_str = Dense(1, name='fc_str')(flat_2)
    fc_thr = Dense(1, name='fc_thr')(flat_2)
    model = Model(inputs=[img_input, vel_input], outputs=[fc_str, fc_thr])
    return model
    
def model_biminet_pretrained(base_model_path):

    base_weightsfile = base_model_path+'.h5'
    base_modelfile   = base_model_path+'.json'
    
    base_json_file = open(base_modelfile, 'r')
    base_loaded_model_json = base_json_file.read()
    base_json_file.close()
    bimi_model = model_from_json(base_loaded_model_json)
    bimi_model.load_weights(base_weightsfile)
    # if config['style_train'] is True:
    bimi_model.trainable = False

    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    ######model#######
    img_input = Input(shape=input_shape)
    lamb_str = Lambda(lambda x: x)(img_input)

    pretrained_model_conv5 = Model(bimi_model.input, bimi_model.get_layer('activation_5').output, name='bimi_model_conv_output')
    pretrained_model_conv5.trainable = False
    base_model_last_output = pretrained_model_conv5([lamb_str])
    flat = Flatten()(base_model_last_output)
    fc_1 = Dense(1000, activation='elu', name='fc_1')(flat)
    fc_2 = Dense(100, activation='elu', name='fc_2')(fc_1)
    fc_3 = Dense(50, activation='elu', name='fc_3')(fc_2)
    fc_4 = Dense(10, activation='elu', name='fc_4')(fc_3)
    fc_out = Dense(config['num_outputs'], name='fc_out')(fc_4)

    model = Model(inputs=[img_input], outputs=[fc_out])
    # model = Model(inputs=[img_input], outputs=[conv_11, z])
    # model = Model(inputs=[img_input, vel_input, delta_input], outputs=[fc_out])
    return model
def model_biminet():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    ######model#######
    img_input = Input(shape=input_shape)
    
    lamb_str = Lambda(lambda x: x)(img_input)
    
    # encoder
    conv_1 = Conv2D(24, (5, 5), padding='same', activation='elu', name='conv2d_1')(lamb_str)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('elu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2), name='pool2d_1')(conv_1)

    conv_2 = Conv2D(36, (5, 5), padding='same', activation='elu', name='conv2d_2')(pool_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('elu')(conv_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2), name='pool2d_2')(conv_2)

    conv_3 = Conv2D(48, (5, 5), padding='same', activation='elu', name='conv2d_3')(pool_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('elu')(conv_3)

    conv_4 = Conv2D(64, (3, 3), padding='same', activation='elu', name='conv2d_4')(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('elu')(conv_4)

    conv_5 = Conv2D(64, (3, 3), padding='same', activation='elu', name='conv2d_5')(conv_4)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation('elu')(conv_5)

    flat = Flatten()(conv_5)
    fc_1 = Dense(1000, activation='elu', name='fc_1')(flat)
    fc_2 = Dense(100 , activation='elu', name='fc_2')(fc_1)
    fc_3 = Dense(50 , activation='elu', name='fc_3')(fc_2)
    fc_4 = Dense(10 , activation='elu', name='fc_4')(fc_3)
    fc_out = Dense(config['num_outputs'], name='fc_out')(fc_4)
    
    model = Model(inputs=[img_input], outputs=[fc_out])
    # model = Model(inputs=[img_input], outputs=[conv_11, z])
    # model = Model(inputs=[img_input, vel_input, delta_input], outputs=[fc_out])
    return model


def model_biminet_latent():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    ######model#######
    img_input = Input(shape=input_shape)
    
    lamb_str = Lambda(lambda x: x)(img_input)
    
    # encoder
    conv_1 = Conv2D(24, (5, 5), padding='same', activation='elu', name='conv2d_1')(lamb_str)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('elu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2), name='pool2d_1')(conv_1)

    conv_2 = Conv2D(36, (5, 5), padding='same', activation='elu', name='conv2d_2')(pool_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('elu')(conv_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2), name='pool2d_2')(conv_2)

    conv_3 = Conv2D(48, (5, 5), padding='same', activation='elu', name='conv2d_3')(pool_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('elu')(conv_3)

    conv_4 = Conv2D(64, (3, 3), padding='same', activation='elu', name='conv2d_4')(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('elu')(conv_4)

    conv_5 = Conv2D(64, (3, 3), padding='same', activation='elu', name='conv2d_5')(conv_4)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation('elu')(conv_5)

    # latent = Flatten()(conv_5)
    # z_mean = Dense(1, activation='relu', name='z_mean')(conv_5)
    # z_log_var = Dense(1, activation='relu', name='z_log_var')(conv_5)

    # z = Lambda(sampling,
    #        output_shape=(latent_dim,), 
    #        name='z')([z_mean, z_log_var])


    # decoder
    conv_6 = Conv2D(64, (3, 3), padding='same', activation='elu', name='conv2d_6')(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation('elu')(conv_6)

    conv_7 = Conv2D(64, (3, 3), padding='same', activation='elu', name='conv2d_7')(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation('elu')(conv_7)

    conc_1 = concatenate([conv_7, conv_4])

    conv_8 = Conv2D(48, (5, 5), padding='same', activation='elu', name='conv2d_8')(conc_1)
    x = BatchNormalization()(conv_8)
    conv_8 = Activation('elu')(x)
    conc_2 = concatenate([conv_8, conv_3])

    upsm_1 = UpSampling2D(size = (2,2), name='upsm2d_1')(conc_2)
    
    conv_9 = Conv2D(36, (5, 5), padding='same', activation='elu', name='conv2d_9')(upsm_1)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation('elu')(conv_9)

    conc_3 = concatenate([conv_9, conv_2])

    upsm_2 = UpSampling2D(size = (2,2), name='upsm2d_2')(conc_3)

    conv_10 = Conv2D(24, (5, 5), padding='same', activation='elu', name='conv2d_10')(upsm_2)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation('elu')(conv_10)
    conc_4 = concatenate([conv_10, conv_1])

    conv_11 = Conv2D(1, (1, 1), activation='sigmoid', name='conv2d_11')(conc_4)

    # fc_1 = Dense(1000, activation='relu', name='fc_1')(flat)
    # fc_2 = Dense(100 , activation='relu', name='fc_2')(fc_1)
    # fc_3 = Dense(50 , activation='relu', name='fc_3')(fc_2)
    # fc_4 = Dense(10 , activation='relu', name='fc_3')(fc_3)
    # fc_out = Dense(config['num_outputs'], name='fc_out')(fc_4)
    
    model = Model(inputs=[img_input], outputs=[conv_11])
    # model = Model(inputs=[img_input], outputs=[conv_11, z])
    # model = Model(inputs=[img_input, vel_input, delta_input], outputs=[fc_out])
    return model


def model_lane_generation_encoder():
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    str_shape = (1,)
    vel_shape = (1,)
    ######model#######
    img_input = Input(shape=img_shape)
    str_input = Input(shape=str_shape)
    vel_input = Input(shape=vel_shape)
    lamb_img = Lambda(lambda x: x)(img_input)
    lamb_str = Lambda(lambda x: x)(str_input)
    lamb_vel = Lambda(lambda x: x)(vel_input)
    
    # encoder
    conv_1 = Conv2D(24, (5, 5), padding='same', activation='elu', name='conv2d_1')(lamb_img)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('elu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2), name='pool2d_1')(conv_1)

    conv_2 = Conv2D(36, (5, 5), padding='same', activation='elu', name='conv2d_2')(pool_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('elu')(conv_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2), name='pool2d_2')(conv_2)

    conv_3 = Conv2D(48, (5, 5), padding='same', activation='elu', name='conv2d_3')(pool_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('elu')(conv_3)

    conv_4 = Conv2D(64, (3, 3), padding='same', activation='elu', name='conv2d_4')(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('elu')(conv_4)

    conv_5 = Conv2D(64, (3, 3), padding='same', activation='elu', name='conv2d_5')(conv_4)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation('elu')(conv_5)

    latent = Flatten()(conv_5)
    fc_s1   = Dense(100)(lamb_str)
    fc_s1   = Activation('elu')(fc_s1)
    fc_s2   = Dense(50)(fc_s1)
    fc_s2   = Activation('elu')(fc_s2)
    fc_v1   = Dense(100)(lamb_vel)
    fc_v1   = Activation('elu')(fc_v1)
    fc_v2   = Dense(50)(fc_v1)
    fc_v2   = Activation('elu')(fc_v2)
    conc_1 = concatenate([latent, fc_s2, fc_v2])
    fc_1   = Dense(100)(conc_1)
    fc_1   = Activation('elu')(fc_1)

    z_mean = Dense(50, activation='elu', name='z_mean')(fc_1)
    z_log_var = Dense(50, activation='elu', name='z_log_var')(fc_1)

    z = Lambda(sampling, name='encoder_output')([z_mean, z_log_var])

    model = Model(inputs=[img_input, str_input, vel_input], outputs=[z_mean, z_log_var, z])
    return model

def model_lane_generation_decoder():
    latent_shape = (50, )
    ######model#######
    latent_input = Input(shape=latent_shape)
    lamb_latent = Lambda(lambda x: x)(latent_input)
    # decoder
    fc_1 = Dense(40*40*64)(lamb_latent)
    fc_1 = Reshape((40, 40, 64))(fc_1)
    ct_1 = Conv2DTranspose(64, 3, activation="elu", strides=2, padding="same")(fc_1)
    ct_2 = Conv2DTranspose(32, 3, activation="elu", strides=2, padding="same")(ct_1)
    ct_3 = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(ct_2)
    
    model = Model(inputs=[latent_input], outputs=[ct_3])
    return model


def sampling(args):
    """Reparameterization trick by sampling 
        fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class NetModel:
    def __init__(self, model_path, base_model_path=None):
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
        
        # if config['vae'] is True:
        #     self._vae_model()
        #     self.encoder = None
        #     self.decoder = None
        # else:
        #     self._model(base_model_path=base_model_path)

    ###########################################################################
    #
    # def _vae_model(self):
    #     self.encoder = model_lane_generation_encoder()
    #     self.decoder = model_lane_generation_decoder()
    #     self.encoder.summary()
    #     self.decoder.summary()
    #     self._compile()
    #     self._vae_compile(self.encoder, self.encoder[1])
    
    # def _vae_compile(self, mu, var):
    #     # learning_rate = config['cnn_lr']
    #     # decay = config['decay']
                      
    #     self.model.compile(optimizer='adam', loss=self._loss_func(mu,var), metrics=["accuracy"])
    #     # self.model.compile(optimizer="rmsprop", loss=self._vae_bce)

    # def _loss_func(encoder_mu, encoder_log_variance):
    #     def vae_reconstruction_loss(y_true, y_predict):
    #         reconstruction_loss_factor = 1000
    #         reconstruction_loss = K.mean(K.square(y_true-y_predict), axis=[1, 2, 3])
    #         return reconstruction_loss_factor * reconstruction_loss

    #     def vae_kl_loss(encoder_mu, encoder_log_variance):
    #         kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=1)
    #         return kl_loss

    #     def vae_kl_loss_metric(y_true, y_predict):
    #         kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=1)
    #         return kl_loss

    #     def vae_loss(y_true, y_predict):
    #         reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
    #         kl_loss = vae_kl_loss(y_true, y_predict)

    #         loss = reconstruction_loss + kl_loss
    #         return loss

    #     return vae_loss

    def _model(self, base_model_path = None):
        if config['network_type'] == const.NET_TYPE_PILOT:
            self.model = model_pilotnet()
        elif config['network_type'] == const.NET_TYPE_BIMI_LATENT:
            self.model = model_biminet_latent()
        elif config['network_type'] == const.NET_TYPE_LATENT_GEN:
            self.model = model_lane_generation_encoder()
        elif config['network_type'] == const.NET_TYPE_BIMI:
            self.model = model_biminet()
        elif config['network_type'] == const.NET_TYPE_BIMI_PRETRAINED:
            self.model = model_biminet_pretrained(base_model_path)
        elif config['network_type'] == const.NET_TYPE_STYLE1:
            self.model = model_style1(base_model_path)
        elif config['network_type'] == const.NET_TYPE_STYLE2:
            self.model = model_style2(base_model_path)
        elif config['network_type'] == const.NET_TYPE_STYLE3:
            self.model = model_style1(base_model_path)
        elif config['network_type'] == const.NET_TYPE_STYLE4:
            self.model = model_style2(base_model_path)
        elif config['network_type'] == const.NET_TYPE_NONLSTM:
            self.model = model_nonlstm()
        else:
            exit('ERROR: Invalid neural network type.')
        self.summary()
        self._compile()

    #
    def _compile(self):
        if config['lstm'] is True:
            learning_rate = config['lstm_lr']
        else:
            learning_rate = config['cnn_lr']
        decay = config['decay']
                      
        if config['latent'] is True:
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
            # self.model.compile(optimizer="rmsprop", loss=self._vae_bce)
        else:
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

        # json_string = self.model.to_json()
        # open(load_model_name+'.json', 'w').write(json_string)
        # self.model = model_from_json(open(load_model_name+'.json').read())
        self.model.load_weights(load_model_name)
        self._compile()
    
    
    def load(self):
        from keras.models import model_from_json
        # self.model = model_from_json(open(self.model_path+'.json').read())
        self.model.load_weights(self.model_path+'.h5')
        self._compile()

    ###########################################################################
    #
    # show summary
    def summary(self):
        self.model.summary()

