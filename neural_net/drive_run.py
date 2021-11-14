#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import numpy as np
from net_model import NetModel
from config import Config
import time, os
import tensorflow as tf
###############################################################################
#

delta = None
standard = None
graph = tf.get_default_graph()
def predict_func(model, np_img, np_img_tb, npvel, name):
    global delta
    global standard
    global graph
    if name == 'delta':
        with graph.as_default():
            # print(np_img.shape)
            delta = model.predict([np_img, np_img_tb, npvel])
        # print(delta[0])
    elif name == 'standard':
        with graph.as_default():
            # print(np_img.shape)
            standard = model.predict([np_img, np_img_tb, npvel])
        # print(standard[0])
    tf.reset_default_graph()
    return
        
class DriveRun:
    
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
    def __init__(self, model_path, delta_model_path):
        
        #self.config = Config()
        self.net_model = NetModel(model_path, delta_model_path)
        self.net_model.load()
   ###########################################################################
    #
    def run(self, input): # input is (image, (vel))
        
        image = input[0]
        image_tb = input[1]
        vel = input[2]
        np_img = np.expand_dims(image, axis=0)
        np_img_tb = np.expand_dims(image_tb, axis=0)
        
        np_img = np_img.reshape(-1, 
                                                Config.neural_net['lstm_timestep'], 
                                                Config.neural_net['input_image_height'],
                                                Config.neural_net['input_image_width'],
                                                Config.neural_net['input_image_depth'])
        np_img_tb = np_img_tb.reshape(-1, 
                                                Config.neural_net['lstm_timestep'], 
                                                Config.neural_net['input_image_height'],
                                                Config.neural_net['input_image_width'],
                                                Config.neural_net['input_image_depth'])
        npvel = np.expand_dims(vel, axis=0).reshape(-1,
                                                    Config.neural_net['lstm_timestep'],
                                                    1)
        
        predict = self.net_model.model.predict([np_img, np_img_tb, npvel])
        steering_angle = predict[0][0]
        steering_angle /= Config.neural_net['steering_angle_scale']
        predict[0][0] = steering_angle

        return predict
    
    
    def run_delta(self, input): # input is (image, (vel))
        from threading import Thread
        
        image = input[0]
        image_tb = input[1]
        vel = input[2]
        # np_img = tf.placeholder('float', shape = [None, Config.neural_net['input_image_height'], Config.neural_net['input_image_width'], Config.neural_net['input_image_depth']]) 
        # np_img_tb = tf.placeholder('float', shape = [None, Config.neural_net['input_image_height'], Config.neural_net['input_image_width'], Config.neural_net['input_image_depth']]) 
        # npvel = tf.placeholder('float', shape = [None, 1]) 
        # np_img_tb = tf.placeholder('float')
                
        np_img = np.expand_dims(image, axis=0)
        np_img_tb = np.expand_dims(image_tb, axis=0)
        
        np_img = np_img.reshape(-1, 
                                                Config.neural_net['lstm_timestep'], 
                                                Config.neural_net['input_image_height'],
                                                Config.neural_net['input_image_width'],
                                                Config.neural_net['input_image_depth'])
        np_img_tb = np_img_tb.reshape(-1, 
                                                Config.neural_net['lstm_timestep'], 
                                                Config.neural_net['input_image_height'],
                                                Config.neural_net['input_image_width'],
                                                Config.neural_net['input_image_depth'])
        npvel = np.expand_dims(vel, axis=0).reshape(-1,
                                                    Config.neural_net['lstm_timestep'],
                                                    1)
        # thread_input = [np_img, np_img_tb, npvel]
        # start = time.time()
        th1 = Thread(target=predict_func, args=(self.net_model.model, np_img, np_img_tb, npvel, 'standard'))
        th2 = Thread(target=predict_func, args=(self.net_model.delta_model, np_img, np_img_tb, npvel, 'delta'))
        th1.start()
        th2.start()
        th1.join()
        th2.join()
        # print("thread : ",time.time()-start)
        # start = time.time()
        # predict = self.net_model.model.predict([np_img, np_img_tb, npvel])
        # delta_predict = self.net_model.delta_model.predict([np_img, np_img_tb, npvel])
        # print("non : ",time.time()-start)
        
        # steering_angle = predict[0][0]
        # steering_angle /= Config.neural_net['steering_angle_scale']
        # predict[0][0] = steering_angle

        return standard, delta
    