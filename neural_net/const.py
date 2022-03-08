#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""
###############################################################################
# constant definition

# network model type

NET_TYPE_PILOT_R    = 1
NET_TYPE_BIMI_R     = 2
NET_TYPE_VGG16_R    = 3
NET_TYPE_ALEX_R     = 4
NET_TYPE_RES_R      = 5
NET_TYPE_DAVE2SKY_R = 6
NET_TYPE_VS_R       = 7
NET_TYPE_CONJOIN_R  = 8
NET_TYPE_PILOTwL_R  = 9

NET_TYPE_PILOT_S    = 11
NET_TYPE_BIMI_S     = 12
NET_TYPE_VGG16_S    = 13
NET_TYPE_ALEX_S     = 14
NET_TYPE_RES_S      = 15
NET_TYPE_DAVE2SKY_S = 16
NET_TYPE_VS_S       = 17
NET_TYPE_CONJOIN_S  = 18
NET_TYPE_PILOTwL_S  = 19

NET_TYPE_ALEXwL_T   = 21

# file extension
DATA_EXT             = '.csv'
IMAGE_EXT            = '.jpg'
LOG_EXT              = '_log.csv'