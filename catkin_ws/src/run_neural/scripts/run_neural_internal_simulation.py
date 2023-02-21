#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
History:

@author: Donghyun Kim
"""

import tensorflow as tf
from tensorflow.keras import layers
# from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, LeakyReLU, Dropout, BatchNormalization, MaxPooling2D, Lambda
from keras.models import Model
from image_process import ImageProcess
import cv2
from config import Config
import time
import rospy
import numpy as np
from std_msgs.msg import Int32, Int32MultiArray
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import math
from image_converter import ImageConverter
from keras.preprocessing.image import array_to_img
import sys
import ros_numpy
from fusion.msg import Control
from keras.backend import clear_session
sys.setrecursionlimit(10**7)
# tf.compat.v1.experimental.output_all_intermediates(True)
# tf.compat.v1.disable_eager_execution()

config = Config.neural_net
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VaeModel():
    def __init__(self, model):

        # gpu_config = tf.compat.v1.ConfigProto()
        # gpu_config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=gpu_config)
        # tf.keras.backend.set_session(self.sess)
        # self.graph = tf.compat.v1.get_default_graph()

        rospy.init_node('vae_run_neural')
        self.intsim_pub = rospy.Publisher('/internal_simulation/image', Image, queue_size=10)
        rospy.Subscriber(Config.data_collection['base_pose_topic'], Image, self._pos_vel_cb)
        rospy.Subscriber(Config.data_collection['vehicle_control_topic'], Control, self._steer_cb)
        rospy.Subscriber(Config.data_collection['camera_image_topic'], Image, self._controller_cb)
        self.rate = rospy.Rate(30)
        
        self.model = model
        # self.model._make_predict_function()
        # self.graph = tf.compat.v1.get_default_graph()
        self.ic = ImageConverter()
        self.image_process = ImageProcess()
        self.velocity = 0.0
        self.steer = 0.0
        self.int_sim_img = Image()

    def _pos_vel_cb(self, value):
        vel_x = value.twist.twist.linear.x 
        vel_y = value.twist.twist.linear.y
        vel_z = value.twist.twist.linear.z
        
        self.velocity = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)

    def _steer_cb(self, value):
        self.steer = value.steer


    def _controller_cb(self, image): 
        img = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        # img = self.ic.imgmsg_to_opencv(image)       
        cropped = img[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                        Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
                        
        img = cv2.resize(cropped, (config['input_image_width'],
                                    config['input_image_height']))
                             
        img = self.image_process.process(img)
        image = np.expand_dims(img, axis=(0,1))
        X_tvel = np.array([[[14]]])
        X_tstr = np.array([[[self.steer]]])
        X_ttime = np.array([[[0.1]]])
        X_train = [image, X_tstr, X_tvel, X_ttime]
        
        # with self.graph.as_default():
        #     with self.session.as_default():
        # with self.graph.as_default():
        x_encoded = self.model.predict(X_train)*255.0
        x_encoded_out = np.squeeze(x_encoded,axis=0)
        pred_image = array_to_img(x_encoded_out)
        pred_image = np.array(pred_image)
        # pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
        # print(pred_image.shape)
        # ros_img = ros_numpy.numpify(pred_image)
        # print(pred_image)
        # self.int_sim_img.data = Int32MultiArray(data=pred_image)
        # self.int_sim_img.data = pred_image
        # self.intsim_pub.publish(self.int_sim_img)
        msg = ros_numpy.msgify(Image, pred_image, encoding='bgr8')
        self.intsim_pub.publish(msg)
        


if __name__ == "__main__":
    try:
        encoder_inputs_img = tf.keras.Input(batch_shape=(1, 1, 160, 160, 3))
        encoder_inputs_str = tf.keras.Input(batch_shape=(1, 1, 1))
        encoder_inputs_vel = tf.keras.Input(batch_shape=(1, 1, 1))
        encoder_inputs_time = tf.keras.Input(batch_shape=(1, 1, 1))
        x = layers.TimeDistributed(layers.Conv2D(24, (5, 5), padding='same', name='conv2d_1'))(encoder_inputs_img)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.Activation('elu'))(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), name='pool2d_1'))(x)
        x = layers.TimeDistributed(layers.Conv2D(36, (5, 5), padding='same', name='conv2d_2'))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.Activation('elu'))(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), name='pool2d_2'))(x)
        x = layers.TimeDistributed(layers.Conv2D(48, (5, 5), padding='same', name='conv2d_3'))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.Activation('elu'))(x)
        x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', name='conv2d_4'))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.Activation('elu'))(x)
        x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', name='conv2d_5'))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.Activation('elu'))(x)
        latent = layers.TimeDistributed(layers.Flatten())(x)
        latent = layers.TimeDistributed(layers.Dense(500))(latent)
        latent = layers.TimeDistributed(layers.LayerNormalization())(latent)
        latent = layers.TimeDistributed(layers.Activation('tanh'))(latent)
        fc_s1  = layers.TimeDistributed(layers.Dense(100))(encoder_inputs_str)
        fc_s1  = layers.TimeDistributed(layers.Activation('elu'))(fc_s1)
        fc_s2  = layers.TimeDistributed(layers.Dense(50))(fc_s1)
        fc_s2  = layers.TimeDistributed(layers.Activation('elu'))(fc_s2)
        fc_v1  = layers.TimeDistributed(layers.Dense(100))(encoder_inputs_vel)
        fc_v1  = layers.TimeDistributed(layers.Activation('elu'))(fc_v1)
        fc_v2  = layers.TimeDistributed(layers.Dense(50))(fc_v1)
        fc_v2  = layers.TimeDistributed(layers.Activation('elu'))(fc_v2)
        fc_t1  = layers.TimeDistributed(layers.Dense(100))(encoder_inputs_time)
        fc_t1  = layers.TimeDistributed(layers.Activation('elu'))(fc_t1)
        fc_t2  = layers.TimeDistributed(layers.Dense(50))(fc_t1)
        fc_t2  = layers.TimeDistributed(layers.Activation('elu'))(fc_t2)
        conc_1 = layers.concatenate([latent, fc_s2, fc_v2, fc_t2])
        bilstm = layers.Bidirectional(layers.LSTM(500, input_shape=(1000, 1), batch_size=1, stateful=True))(conc_1)
        fc_1   = layers.Dense(500)(bilstm)
        fc_1   = layers.Activation('elu')(fc_1)
        fc_2   = layers.Dense(100)(fc_1)
        x   = layers.Activation('elu')(fc_2)
        z_mean = layers.Dense(50, name="z_mean")(x)
        z_log_var = layers.Dense(50, name="z_log_var")(x)
        encoder_output = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model([ encoder_inputs_img, encoder_inputs_str,
                                encoder_inputs_vel, encoder_inputs_time], 
                                [z_mean, z_log_var, encoder_output], 
                                name="encoder")
        ## Decoder
        latent_inputs = tf.keras.Input(shape=(50,))
        x = layers.Dense(40 * 40 * 64, activation="elu")(latent_inputs)
        x = layers.Reshape((40, 40, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="elu",  padding="same")(x)
        x = layers.Conv2DTranspose(36, 3, activation="elu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(24, 3, activation="elu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
        ## VAE
        model_input = [ encoder_inputs_img, encoder_inputs_str,
                                encoder_inputs_vel, encoder_inputs_time]
        model_output = decoder(encoder_output)
        VarAE=Model(model_input, model_output)
        # VarAE.summary()
        pretrained_path = '/home2/kdh/av/latent/vae_batch10_lstm/vae_kdh_batch10_timestep1'
        weightsfile = pretrained_path+'.h5'
        VarAE.load_weights(weightsfile)
        v = VaeModel(VarAE)

        while not rospy.is_shutdown():
            # v.intsim_pub.publish(v.int_sim_img)
            v.rate.sleep()
        
    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
        
