#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: Donghyun Kim
# -*- coding: utf-8 -*-
"""

import keras
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


class VAE(keras.Model):
    def __init__(self, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = self.model_lane_generation_encoder()
        self.decoder = self.model_lane_generation_decoder()
        # self.total_loss_tracker = tf.metrics.mean(name="total_loss")
        # self.reconstruction_loss_tracker = tf.metrics.mean(
        #     name="reconstruction_loss"
        # )
        # self.kl_loss_tracker = tf.metrics.mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def model_lane_generation_encoder(self):
        img_shape = (config['input_image_height'],
                        config['input_image_width'],
                        config['input_image_depth'],)
        str_shape = (1,)
        vel_shape = (1,)
        time_shape = (1,)
        ######model#######
        img_input = Input(shape=img_shape)
        str_input = Input(shape=str_shape)
        vel_input = Input(shape=vel_shape)
        time_input = Input(shape=time_shape)
        lamb_img = Lambda(lambda x: x)(img_input)
        lamb_str = Lambda(lambda x: x)(str_input)
        lamb_vel = Lambda(lambda x: x)(vel_input)
        lamb_time = Lambda(lambda x: x)(time_input)
        
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
        fc_t1   = Dense(100)(lamb_time)
        fc_t1   = Activation('elu')(fc_t1)
        fc_t2   = Dense(50)(fc_t1)
        fc_t2   = Activation('elu')(fc_t2)
        conc_1 = concatenate([latent, fc_s2, fc_v2, fc_t2])
        fc_1   = Dense(100)(conc_1)
        fc_1   = Activation('elu')(fc_1)

        z_mean = Dense(50, activation='elu', name='z_mean')(fc_1)
        z_log_var = Dense(50, activation='elu', name='z_log_var')(fc_1)

        z = Lambda(self.sampling, name='encoder_output')([z_mean, z_log_var])

        model = Model(inputs=[img_input, str_input, vel_input, time_input], outputs=[z_mean, z_log_var, z])
        return model

    def model_lane_generation_decoder(self):
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

    def sampling(self, args):
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


# class NetModel:
#     def __init__(self, model_path, base_model_path=None):
#         self.model = None
#         model_name = model_path[model_path.rfind('/'):] # get folder name
#         self.name = model_name.strip('/')

#         self.model_path = model_path
#         #self.config = Config()

#         # to address the error:
#         #   Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
#         os.environ["CUDA_VISIBLE_DEVICES"]=str(config['gpus'])
        
#         gpu_options = tf.GPUOptions(allow_growth=True)
#         sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#         K.tensorflow_backend.set_session(sess)
#         # with tf.device('/cpu:0'):
        
#         if config['vae'] is True:
#             self._vae_model()
#             self.encoder = None
#             self.decoder = None
#         else:
#             self._model(base_model_path=base_model_path)

#     ###########################################################################
#     #
#     def _vae_model(self):
#         self.encoder = model_lane_generation_encoder()
#         self.decoder = model_lane_generation_decoder()
#         self.encoder.summary()
#         self.decoder.summary()
#         self._compile()
#         self._vae_compile(self.encoder, self.encoder[1])
    
#     def _vae_compile(self, mu, var):
#         # learning_rate = config['cnn_lr']
#         # decay = config['decay']
                      
#         self.model.compile(optimizer='adam', loss=self._loss_func(mu,var), metrics=["accuracy"])
#         # self.model.compile(optimizer="rmsprop", loss=self._vae_bce)

#     def _loss_func(encoder_mu, encoder_log_variance):
#         def vae_reconstruction_loss(y_true, y_predict):
#             reconstruction_loss_factor = 1000
#             reconstruction_loss = K.mean(K.square(y_true-y_predict), axis=[1, 2, 3])
#             return reconstruction_loss_factor * reconstruction_loss

#         def vae_kl_loss(encoder_mu, encoder_log_variance):
#             kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=1)
#             return kl_loss

#         def vae_kl_loss_metric(y_true, y_predict):
#             kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=1)
#             return kl_loss

#         def vae_loss(y_true, y_predict):
#             reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
#             kl_loss = vae_kl_loss(y_true, y_predict)

#             loss = reconstruction_loss + kl_loss
#             return loss

#         return vae_loss

#     def _model(self, base_model_path = None):
#         if config['network_type'] == const.NET_TYPE_PILOT:
#             self.model = model_pilotnet()
#         elif config['network_type'] == const.NET_TYPE_BIMI_LATENT:
#             self.model = model_biminet_latent()
#         elif config['network_type'] == const.NET_TYPE_LATENT_GEN:
#             self.model = model_lane_generation_encoder()
#         elif config['network_type'] == const.NET_TYPE_BIMI:
#             self.model = model_biminet()
#         elif config['network_type'] == const.NET_TYPE_BIMI_PRETRAINED:
#             self.model = model_biminet_pretrained(base_model_path)
#         elif config['network_type'] == const.NET_TYPE_STYLE1:
#             self.model = model_style1(base_model_path)
#         elif config['network_type'] == const.NET_TYPE_STYLE2:
#             self.model = model_style2(base_model_path)
#         elif config['network_type'] == const.NET_TYPE_STYLE3:
#             self.model = model_style1(base_model_path)
#         elif config['network_type'] == const.NET_TYPE_STYLE4:
#             self.model = model_style2(base_model_path)
#         elif config['network_type'] == const.NET_TYPE_NONLSTM:
#             self.model = model_nonlstm()
#         else:
#             exit('ERROR: Invalid neural network type.')
#         self.summary()
#         self._compile()

#     #
#     def _compile(self):
#         if config['lstm'] is True:
#             learning_rate = config['lstm_lr']
#         else:
#             learning_rate = config['cnn_lr']
#         decay = config['decay']
                      
#         if config['latent'] is True:
#             self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
#             # self.model.compile(optimizer="rmsprop", loss=self._vae_bce)
#         else:
#             self.model.compile(loss=losses.mean_squared_error,
#                     optimizer=optimizers.Adam(lr=learning_rate, decay=decay, clipvalue=1), 
#                     metrics=['accuracy'])


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

