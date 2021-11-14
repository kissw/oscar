#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Donghyun Kim
"""


import sys
import os
import pandas as pd
from numpy.core.numeric import full
from progressbar import ProgressBar

import const
from drive_data import DriveData
from config import Config

###############################################################################
#
def search(dir_name):
    filenames = os.listdir(dir_name)
    csv_filenames = []
    for filename in filenames:
        if const.DATA_EXT in filename :
            full_filename = os.path.join(dir_name, filename)
            csv_filenames.append(full_filename)
            # print(csv_filenames)
    return csv_filenames
        

def append_csv(data_path):
    filenames = search(data_path)
    print(filenames)
    # add '/' at the end of data_path if user doesn't specify
    if data_path[-1] != '/':
        data_path = data_path + '/'

    # find the second '/' from the end to get the folder name
    loc_dir_delim = data_path[:-1].rfind('/')
    if (loc_dir_delim != -1):
        folder_name = data_path[loc_dir_delim+1:-1]
        csv_file = folder_name + '_all' + const.DATA_EXT
    else:
        folder_name = data_path[:-1]
        csv_file = folder_name + '_all' + const.DATA_EXT

    new_csv = []
    # print(data_path)
    for filename in filenames:
        # print(filename)
        data = DriveData(filename)
        data.read(normalize = False)

        # check image exists
        bar = ProgressBar()
        for i in bar(range(len(data.df))):
            if os.path.exists(data_path + data.image_names[i]):
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
                            + str(data.velocities_xyz[i][3]) + ','
                            + str(data.velocities_xyz[i][4]) + ','
                            + str(data.delta[i][0]) + ','
                            + str(data.delta[i][1]) + ','
                            + str(data.delta[i][2]) + '\n')


    # write a new csv
    new_csv_fh = open(data_path + csv_file, 'w')
    for i in range(len(new_csv)):
        new_csv_fh.write(new_csv[i])
    new_csv_fh.close()
    data = pd.read_csv(data_path + csv_file)
    # data = data.sort_values(by=)
    # print(data.head())

###############################################################################
#
def main():
    if (len(sys.argv) != 2):
        print('Usage: \n$ python append_csv.py data_folder_name')
        return

    append_csv(sys.argv[1])


###############################################################################
#
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nShutdown requested. Exiting...')
