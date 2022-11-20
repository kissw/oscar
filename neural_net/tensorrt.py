#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:49:23 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock, donghyun
"""

import sys, os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np
from PIL import Image
from vis.utils import utils
from vis.visualization import visualize_cam, visualize_saliency
from keras.models import Sequential, Model

from drive_run import DriveRun
from drive_data import DriveData
from config import Config
from image_process import ImageProcess


def main(model_path, image_file_path):
    image_process = ImageProcess()

    image = cv2.imread(image_file_path)

    # if collected data is not cropped then crop here
    # otherwise do not crop.
    if Config.data_collection['crop'] is not True:
        image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                      Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]

    image = cv2.resize(image, 
                        (Config.neural_net['input_image_width'],
                         Config.neural_net['input_image_height']))
    image = image_process.process(image)

    drive_run = DriveRun(model_path)
    measurement = drive_run.run((image, ))


###############################################################################
#       
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 2):
            exit('Usage:\n$ python {} model_path'.format(sys.argv[0]))

        main(sys.argv[1])
        # images_saliency(sys.argv[1], sys.argv[2], Config.neural_net['lstm'])
        # show_layer_saliency(sys.argv[1], sys.argv[2])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
