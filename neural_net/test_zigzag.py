#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:49:23 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
from drive_run import DriveRun
from config import Config
from image_process import ImageProcess

config = Config.neural_net
image_process = ImageProcess()


###############################################################################
#
def main(model_path):
    # image_file_name1 = '/mnt/Data/oscar/kics/normal/2023-01-12-18-50-55/train/2023-01-12-18-50-55/2023-01-12-18-52-46-750403.jpg'
    # image_file_name2 = '/mnt/Data/oscar/kics/normal/2023-01-12-18-50-55/train/2023-01-12-18-50-55/2023-01-12-18-52-47-750782.jpg'
    # image_file_name3 = '/mnt/Data/oscar/kics/normal/2023-01-12-18-50-55/train/2023-01-12-18-50-55/2023-01-12-18-52-48-756956.jpg'
    # image_file_name4 = '/mnt/Data/oscar/kics/normal/2023-01-12-18-50-55/train/2023-01-12-18-50-55/2023-01-12-18-52-49-753387.jpg'
    # image_file_name5 = '/mnt/Data/oscar/kics/normal/2023-01-12-18-50-55/train/2023-01-12-18-50-55/2023-01-12-18-52-50-753219.jpg'
    # steering1 = -0.118721820414
    # steering2 = -0.175839617848
    # steering3 = -0.188336148858
    # steering4 = -0.186376541853
    # steering5 = -0.161961734295
    image_file_name1 = '/mnt/Data/oscar/kics/normal/2023-01-12-18-50-54/train/2023-01-12-18-50-54/2023-01-12-18-56-10-218984.jpg'
    image_file_name2 = '/mnt/Data/oscar/kics/normal/2023-01-12-18-50-54/train/2023-01-12-18-50-54/2023-01-12-18-56-11-219472.jpg'
    image_file_name3 = '/mnt/Data/oscar/kics/normal/2023-01-12-18-50-54/train/2023-01-12-18-50-54/2023-01-12-18-56-12-219768.jpg'
    image_file_name4 = '/mnt/Data/oscar/kics/normal/2023-01-12-18-50-54/train/2023-01-12-18-50-54/2023-01-12-18-56-13-220101.jpg'
    image_file_name5 = '/mnt/Data/oscar/kics/normal/2023-01-12-18-50-54/train/2023-01-12-18-50-54/2023-01-12-18-56-14-220528.jpg'
    steering1 = 0.00464684469625
    steering2 = 0.0122282849625
    steering3 = 0.157657012343
    steering4 = 0.0
    steering5 = 0.0299932695925

    imgs_path = []
    steers = []
    imgs_path.append((image_file_name1, image_file_name2, image_file_name3, image_file_name4, image_file_name5))
    steers.append((steering1, steering2, steering3, steering4, steering5))

    inputs_images = []
    inputs_strs = []
    for i in range(len(imgs_path[0])):
        # print(imgs_path[0][i])
        img = cv2.imread(imgs_path[0][i])
        img = img[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                      Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
        img = cv2.resize(img, (Config.neural_net['input_image_width'],
                    Config.neural_net['input_image_height']))
        img = image_process.process(img)
        
        inputs_images.append(img)
        inputs_strs.append(steers[0][i])
        # print(steers[0][i])

    # print(inputs_strs)
    X_train_img = np.expand_dims(np.array(inputs_images), axis=0)
    X_train_str = np.array(inputs_strs).reshape(-1,config['lstm_timestep'],1)
    print(X_train_img.shape)
    print(X_train_str.shape)
    X_train = [X_train_img, X_train_str]


    drive_run = DriveRun(model_path)
    predict = drive_run.run(X_train)
    # zigzag = predict[0][0]

    print("{:.2f}".format(float(predict)))

###############################################################################
#       
if __name__ == '__main__':
    try:
        if len(sys.argv) == 2:
            main(sys.argv[1])
        else:
            exit('Usage:\n$ python {} model_path'.format(sys.argv[0]))


    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
