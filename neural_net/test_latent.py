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

from latent_run import LatentRun
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

    latent_run = LatentRun(model_path)
    measurement = latent_run.run((image, ))
    print(np.squeeze(measurement))
    plt.figure()
    #plt.title('Saliency Visualization' + str(measurement))
    # plt.title('Steering Angle Prediction: ' + str(measurement[0][0]))
    # layer_idx = utils.find_layer_idx(latent_run.net_model.model, 'conv2d_last')
    # heatmap = visualize_cam(latent_run.net_model.model, layer_idx, 
    #             filter_indices=None, seed_input=image, backprop_modifier='guided')

    # plt.imshow(measurement)
    plt.imshow(np.squeeze(measurement), cmap='gray', vmin=0, vmax=1)
    plt.show()

    plt.imshow(image)
    plt.show()
    # plt.imshow(heatmap, cmap='jet', alpha=0.5)

    # file name
    # loc_slash = image_file_path.rfind('/')
    # if loc_slash != -1: # there is '/' in the data path
    #     image_file_name = image_file_path[loc_slash+1:] 

    # saliency_file_path = model_path + '_' + image_file_name + '_saliency.png'
    # saliency_file_path_pdf = model_path + '_' + image_file_name + '_saliency.pdf'

    # plt.tight_layout()
    # # save fig    
    # plt.savefig(saliency_file_path, dpi=150)
    # plt.savefig(saliency_file_path_pdf, dpi=150)

    # print('Saved ' + saliency_file_path +' & .pdf')

    # show the plot 
    #plt.show()

###############################################################################
#       
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 3):
            exit('Usage:\n$ python {} model_path, image_file_name'.format(sys.argv[0]))

        main(sys.argv[1], sys.argv[2])
        
        # images_cam(sys.argv[1], sys.argv[2])
        # images_saliency(sys.argv[1], sys.argv[2], Config.neural_net['lstm'])
        # show_layer_saliency(sys.argv[1], sys.argv[2])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
