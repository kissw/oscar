#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 16:18:14 2022
History:

@author: Donghyun Kim
"""

#####
# We don't need this cropping process anymore
#####

from PIL import Image
import os, sys, shutil, cv2
import const
import random, math
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
    print(data_path + folder_name + '_imgen' + const.DATA_EXT)
    
    data = DriveData(csv_backup_name)
    data.read(normalize = False)
    
    num_data = len(data.df)
    # image_names = []
    # try:
    #     if not os.path.exists(data_path + 'seg'):
    #         os.makedirs(data_path + 'seg') 
    # except OSError:
    #     print("Error: Failed to create the directory.")        

    bar = ProgressBar()

    new_csv = []
    for i in bar(range(1, num_data-1)): # we don't have a title
        # print(i)
        current_image_path = data_path + data.df.loc[i]['image_fname']
        
        # for frame in range(20):
        for n in range(10):
            frame = random.randrange(20)+1
            # while i+frame >= num_data or i-frame < 0:
            # frame : 1~20
            # i     : 0~710
            # numda : 711
            while i+frame >= num_data:
                frame = random.randrange(20)+1
            # 710, 1
            # print(i, frame)
            v = 0
            s = 0
            t = 0
            v_i = 0
            s_i = 0
            t_i = 0
            for f in range(0, frame+1):
                v_i += float(data.df.loc[i+f]['vel'])
                s_i += float(data.df.loc[i+f]['steering_angle'])
                # t_i += float(data.df.loc[i+f]['linux_time'])
            v = v_i / (frame+1)
            s = s_i / (frame+1)
            t = float(data.df.loc[i+frame]['linux_time'])- float(data.df.loc[i]['linux_time'])
            
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
                        + str(data.df.loc[i+frame]['image_fname']) + ','
                        + str(float(s)) + ','
                        + str(float(v)) + ',' 
                        + str(float(t)) + ','
                        + str(data.image_names[i-1]) + '\n')

    # write a new csv
    new_csv_fh = open(data_path + folder_name + '_imgen' + const.DATA_EXT, 'w')
    for i in range(len(new_csv)):
        new_csv_fh.write(new_csv[i])
    new_csv_fh.close()

###############################################################################
#       
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 2):
            exit('Usage:\n$ data_path')
        
        main(sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
       
