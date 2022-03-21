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

NET_TYPE_PILOT_S    = 1
NET_TYPE_BIMI_S     = 2
NET_TYPE_VGG16_S    = 3
NET_TYPE_ALEX_S     = 4
NET_TYPE_RES_S      = 5
NET_TYPE_DAVE2SKY_S = 6
NET_TYPE_VS_S       = 7
NET_TYPE_CONJOIN_S  = 8

NET_TYPE_PILOT_R    = 11
NET_TYPE_BIMI_R     = 12
NET_TYPE_VGG16_R    = 13
NET_TYPE_ALEX_R     = 14
NET_TYPE_RES_R      = 15
NET_TYPE_DAVE2SKY_R = 16
NET_TYPE_VS_R       = 17
NET_TYPE_CONJOIN_R  = 18

NET_TYPE_PILOTwL_S  = 21
NET_TYPE_BIMIwL_S   = 22
NET_TYPE_ALEXwL_S   = 24
NET_TYPE_RESwL_S    = 25
NET_TYPE_CONJOINwL_S= 28

NET_TYPE_PILOTwL_R  = 31
NET_TYPE_BIMIwL_R   = 32
NET_TYPE_ALEXwL_R   = 34
NET_TYPE_RESwL_R    = 35
NET_TYPE_CONJOINwL_R= 38

NET_TYPE_PILOT_S_3  = 41
NET_TYPE_PILOT_S_7  = 42
NET_TYPE_PILOT_S_9  = 43

# file extension
DATA_EXT             = '.csv'
IMAGE_EXT            = '.jpg'
LOG_EXT              = '_log.csv'