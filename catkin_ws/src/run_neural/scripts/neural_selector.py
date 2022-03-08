#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for DST 

@author: Donghyun
"""

import threading 
import cv2
import time
import rospy
import numpy as np
from std_msgs.msg import Int32, Bool
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

import sys
import os

from config import Config

if Config.data_collection['vehicle_name'] == 'fusion':
    from fusion.msg import Control
elif Config.data_collection['vehicle_name'] == 'rover':
    from geometry_msgs.msg import Twist
    from rover.msg import Control
else:
    exit(Config.data_collection['vehicle_name'] + 'not supported vehicle.')


config = Config.neural_net
class NeuralSelector:
    def __init__(self):
        rospy.init_node('neural_selector')
        self.rate = rospy.Rate(30)
        self.network = 0
        self.nerual1_joy_data = Control()
        self.nerual2_joy_data = Control()
        rospy.Subscriber('/neural_select', Int32, self.neural_select_cb)
        rospy.Subscriber(Config.data_collection['vehicle_control_topic1'], Control, self.neural1_cb)
        rospy.Subscriber(Config.data_collection['vehicle_control_topic2'], Control, self.neural2_cb)
        self.joy_pub = rospy.Publisher(Config.data_collection['vehicle_control_topic'], Control, queue_size = 10)

    def neural_select_cb(self, value):
        if value.data is 1:
            self.network = 1
            print('Network 1 selected')
        elif value.data is 2:
            self.network = 2
            print('Network 2 selected')
        else:
            self.network = 0

    def neural1_cb(self, value):
        self.nerual1_joy_data.steer = value.steer
        self.nerual1_joy_data.throttle = value.throttle
        self.nerual1_joy_data.brake = value.brake
        
    def neural2_cb(self, value):
        self.nerual2_joy_data.steer = value.steer
        self.nerual2_joy_data.throttle = value.throttle
        self.nerual2_joy_data.brake = value.brake
            
    def main(self):
        if self.network is 1:
            self.joy_pub.publish(self.nerual1_joy_data)
        elif self.network is 2:
            self.joy_pub.publish(self.nerual2_joy_data)

        self.rate.sleep()

if __name__ == "__main__":
    try:
        ns = NeuralSelector()
        while not rospy.is_shutdown():
            ns.main()

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
        
