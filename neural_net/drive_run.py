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
###############################################################################
#
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
        vel = input[1]
        np_img = np.expand_dims(image, axis=0)
        
        np_img = np_img.reshape(-1, 
                                                Config.neural_net['lstm_timestep'], 
                                                Config.neural_net['input_image_height'],
                                                Config.neural_net['input_image_width'],
                                                Config.neural_net['input_image_depth'])
        npvel = np.expand_dims(vel, axis=0).reshape(-1,
                                                    Config.neural_net['lstm_timestep'],
                                                    1)
        
        predict = self.net_model.model.predict([np_img, npvel])
        delta_predict = self.net_model.delta_model.predict([np_img, npvel])
        steering_angle = predict[0][0]
        steering_angle /= Config.neural_net['steering_angle_scale']
        predict[0][0] = steering_angle

        return predict, delta_predict