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
import time
from keras.models import Model
###############################################################################
#
class DriveRun:
    
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
    def __init__(self, model_path, base_weight_file = None):
        
        #self.config = Config()
        self.net_model = NetModel(model_path, base_weight_file)
        # self.net_model_base = NetModel(base_weight_file)
        self.net_model.load()
        # self.net_model_base.load()

   ###########################################################################
    #
    def run(self, input): # input is (image, (vel))
        if Config.neural_net['style_run'] is False:
            image = input[0]
            if Config.neural_net['num_inputs'] == 2:
                velocity = input[1]
            np_img = np.expand_dims(image, axis=0)
            
            if Config.neural_net['num_inputs'] == 2:
                velocity = np.array(velocity).reshape(-1, 1)
                predict = self.net_model.model.predict([np_img, velocity])
                # steer = self.net_model.model.get_layer('fc_')
            else:
                predict = self.net_model.model.predict(np_img)
            # calc scaled steering angle
            steering_angle = predict[0][0]
            throttle = predict[0][1]
            steering_angle /= Config.neural_net['steering_angle_scale']
            throttle /= Config.neural_net['throttle_scale']
            predict[0][0] = steering_angle
            predict[0][1] = throttle
            if Config.neural_net['num_outputs'] == 3:
                brake = predict[0][2]
                brake /= Config.neural_net['brake_scale']
                predict[0][2] = brake
            
            return predict
                
        else:
            image = input[0]
            # base_model_image = input[1]
            velocity = input[1]
            goal_velocity = input[2]
            
            np_img = np.expand_dims(image, axis=0)
            # np_base_model_img = np.expand_dims(base_model_image, axis=0)
            np_vel = np.array(velocity).reshape(-1, 1)
            np_goalvel = np.array(goal_velocity).reshape(-1, 1)
            predict = self.net_model.model.predict([np_img, np_vel, np_goalvel])
            if Config.neural_net['only_thr_brk'] is True:
                steering_angle = self.net_model.base_model.predict([np_img, np_vel])[0][0]
                throttle = predict[0][0]
                brake = predict[0][1]
            else:
                steering_angle = predict[0][0]
                throttle = predict[0][1]
                brake = predict[0][2]
                
            # brake = 0
            
            steering_angle /= Config.neural_net['steering_angle_scale']
            throttle /= Config.neural_net['throttle_scale']
            brake /= Config.neural_net['brake_scale']
            if throttle < 0:
                throttle = 0
            if brake < 0:
                brake = 0
            # predict[0][0] = steering_angle
            # predict[0][1] = throttle
            # predict[0][2] = brake
            return steering_angle, throttle, brake
        # print(brake)