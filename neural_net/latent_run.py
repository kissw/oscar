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
class LatentRun:
    
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
    def __init__(self, model_path, base_weight_file = None):
        
        #self.config = Config()
        self.net_model = NetModel(model_path, base_weight_file)
        self.net_model.load()
        print('Done')

   ###########################################################################
    #
    def run(self, input): # input is (image, (vel))
        
        image = input[0]
        np_img = np.expand_dims(image, axis=0)
        
        predict = self.net_model.model.predict(np_img)
        #
        return predict