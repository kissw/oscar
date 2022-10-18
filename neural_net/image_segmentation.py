#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

#####
# We don't need this cropping process anymore
#####

from PIL import Image
import os, sys, shutil, cv2
import const
from progressbar import ProgressBar
from drive_data import DriveData
from config import Config

###############################################################################
#       
def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith(extension))

###############################################################################
#       
def main(data_path):
    if data_path[-1] != '/':
        data_path = data_path + '/'

    # find the second '/' from the end to get the folder name
    loc_dir_delim = data_path[:-1].rfind('/')
    if (loc_dir_delim != -1):
        folder_name = data_path[loc_dir_delim+1:-1]
        csv_file = folder_name + const.DATA_EXT
    else:
        folder_name = data_path[:-1]
        csv_file = folder_name + const.DATA_EXT

    csv_backup_name = data_path + csv_file + '.bak'
    # print(csv_backup_name)
    shutil.copy2(data_path + csv_file, csv_backup_name)
    # os.rename(data_path + csv_file, csv_backup_name)
    print('rename ' + data_path + csv_file + ' to ' + csv_backup_name)
    
    data = DriveData(csv_backup_name)
    data.read(normalize = False)
    
    num_data = len(data.df)
    # image_names = []
    segimg_names = []

    lower_yellow = (24, 60, 60)
    upper_yellow = (100, 255, 255)

    bar = ProgressBar()
    for i in bar(range(num_data)): # we don't have a title
        image_path = data_path + data.df.loc[i]['image_fname']
        
        image = cv2.imread(image_path)

        cropped = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                            Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
        
        image = cv2.resize(cropped, 
                                (Config.neural_net['input_image_width'],
                                Config.neural_net['input_image_height']))

        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
        img_result = cv2.bitwise_and(image, image, mask=img_mask)

        cv2.imwrite(data_path + 'seg/' + data.df.loc[i]['image_fname'][:-4] + '_latent' + const.IMAGE_EXT, img_result)
        segimg_names.append(str(data.df.loc[i]['image_fname'][:-4] + '_latent' + const.IMAGE_EXT))

    new_csv = []
    bar = ProgressBar()
    for i in bar(range(len(data.df))):
        if os.path.exists(data_path + data.image_names[i]):
            if Config.data_collection['brake'] is True:
                new_csv.append(data.image_names[i] + ','
                            + str(data.measurements[i][0]) + ','
                            + str(data.measurements[i][1]) + ','
                            + str(data.measurements[i][2]) + ',' # brake
                            + str(data.time_stamps[i]) + ','
                            + str(data.velocities[i]) + ','
                            + str(data.velocities_xyz[i][0]) + ','
                            + str(data.velocities_xyz[i][1]) + ','
                            + str(data.velocities_xyz[i][2]) + ','
                            + str(data.positions_xyz[i][0]) + ','
                            + str(data.positions_xyz[i][1]) + ','
                            + str(data.positions_xyz[i][2]) + ','
                            + str(segimg_names[i]) + '\n')
            else:
                new_csv.append(data.image_names[i] + ','
                            + str(data.measurements[i][0]) + ','
                            + str(data.measurements[i][1]) + ','
                            + str(data.time_stamps[i]) + ','
                            + str(data.velocities[i]) + ','
                            + str(data.velocities_xyz[i][0]) + ','
                            + str(data.velocities_xyz[i][1]) + ','
                            + str(data.velocities_xyz[i][2]) + ','
                            + str(data.positions_xyz[i][0]) + ','
                            + str(data.positions_xyz[i][1]) + ','
                            + str(data.positions_xyz[i][2]) + ','
                            + str(segimg_names[i]) + '\n')

    # write a new csv
    new_csv_fh = open(data_path + folder_name + '_latent' + const.DATA_EXT, 'w')
    for i in range(len(new_csv)):
        new_csv_fh.write(new_csv[i])
    new_csv_fh.close()

###############################################################################
#       
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 2):
            exit('Usage:\n$ image_crop data_path')
        
        main(sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
       
