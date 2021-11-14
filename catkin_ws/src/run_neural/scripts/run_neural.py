#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 
@author: jaerock
"""

import threading 
import cv2
import time
import rospy
import numpy as np
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import math

import sys
import os

import const
from image_converter import ImageConverter
from drive_run import DriveRun
from config import Config
from image_process import ImageProcess
from tf.transformations import euler_from_quaternion, quaternion_from_euler

if Config.data_collection['vehicle_name'] == 'fusion':
    from fusion.msg import Control
elif Config.data_collection['vehicle_name'] == 'rover':
    from geometry_msgs.msg import Twist
    from rover.msg import Control
else:
    exit(Config.data_collection['vehicle_name'] + 'not supported vehicle.')


config = Config.neural_net
velocity = 0.0
class NeuralControl:
    def __init__(self, model_path, delta_model_path):
        rospy.init_node('run_neural')
        self.ic = ImageConverter()
        self.image_process = ImageProcess()
        self.rate = rospy.Rate(30)
        self.drive= DriveRun(model_path, delta_model_path)
        rospy.Subscriber(Config.data_collection['camera_image_topic'], Image, self._controller_cb)
        self.image = None
        self.image_crop = None
        self.image_processed = False
        #self.config = Config()
        self.braking = False
        self.lstm_image = []
        self.lstm_image_tb = []
        self.lstm_vel = []
        self.term_count = 0
    def _controller_cb(self, image): 
        img = self.ic.imgmsg_to_opencv(image)
        cropped = img[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                      Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
                      
        img = cv2.resize(cropped, (config['input_image_width'],
                                   config['input_image_height']))
                                          
        # origin_image = self.image_process.process(img)
        
        crop_tb = img[0:30, 0:159]
        crop_str = img[31:159, 0:159]
        crop_tb = cv2.resize(crop_tb, 
                            (config['input_image_width'],
                            config['input_image_height']))
        crop_str = cv2.resize(crop_str, 
                            (config['input_image_width'],
                            config['input_image_height']))
        
        crop_tb = self.image_process.process(crop_tb)
        crop_str = self.image_process.process(crop_str)
        
        # self.image = origin_image
        
        if self.term_count % Config.run_neural['lstm_dataterm'] is 0:
            self.lstm_image.append(crop_str)
            self.lstm_image_tb.append(crop_tb)
            self.lstm_vel.append(velocity)
            if len(self.lstm_image) > config['lstm_timestep'] :
                del self.lstm_image[0]
                del self.lstm_image_tb[0]
                del self.lstm_vel[0]
        self.term_count += 1
                    
        self.image_processed = True
        
    def _timer_cb(self):
        self.braking = False

    def apply_brake(self):
        self.braking = True
        timer = threading.Timer(Config.run_neural['brake_apply_sec'], self._timer_cb) 
        timer.start()


def pos_vel_cb(value):
    global velocity

    vel_x = value.twist.twist.linear.x 
    vel_y = value.twist.twist.linear.y
    vel_z = value.twist.twist.linear.z
    
    velocity = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)

def main(model_path):
    
    # ready for neural network
    neural_control = NeuralControl(model_path, model_path)
    
    rospy.Subscriber(Config.data_collection['base_pose_topic'], Odometry, pos_vel_cb)
    # ready for /bolt topic publisher
    joy_pub = rospy.Publisher(Config.data_collection['vehicle_control_topic'], Control, queue_size = 10)
    joy_data = Control()
    pred_data = Control()

    if Config.data_collection['vehicle_name'] == 'rover':
        joy_pub4mavros = rospy.Publisher(Config.config['mavros_cmd_vel_topic'], Twist, queue_size=20)

    print('\nStart running. Vroom. Vroom. Vroooooom......')
    # print('steer \tthrt: \tbrake \tvelocity \tHz')
    print('steer \td_str \tthrt \t d_thr \tbrake \tvelocity \tHz')

    use_predicted_throttle = True if config['num_outputs'] == 2 else False
    while not rospy.is_shutdown():

        if neural_control.image_processed is False:
            continue
        
        start = time.time()
        end = time.time()
        # predicted steering angle from an input image
                
        if len(neural_control.lstm_vel) >= config['lstm_timestep']:
            prediction = neural_control.drive.run((neural_control.lstm_image, neural_control.lstm_image_tb, neural_control.lstm_vel))
            
            pred_data.steer     = prediction[0][0][0]
            pred_data.throttle  = prediction[1][0][0]
            pred_data.brake     = prediction[2][0][0]
            if pred_data.throttle < 0 :
                pred_data.throttle = 0
            elif pred_data.throttle > 1 :
                pred_data.throttle = 1
            
            if pred_data.brake < 0 :
                pred_data.brake = 0
            elif pred_data.brake > 1 :
                pred_data.brake = 1
                
            
            joy_data.steer = (pred_data.steer) * Config.run_neural['scale_factor_steering']
            joy_data.throttle = (pred_data.throttle) * Config.run_neural['scale_factor_throttle']
            joy_data.brake = pred_data.brake
        
        if joy_data.throttle < 0 :
            joy_data.throttle = 0
        elif joy_data.throttle > 1 :
            joy_data.throttle = 1
        
        if joy_data.brake < 0 :
            joy_data.brake = 0
        elif joy_data.brake > 1 :
            joy_data.brake = 1
            
        joy_pub.publish(joy_data)

        ## print out
        # print(joy_data.steer, joy_data.throttle, joy_data.brake, velocity)
        # cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f} \t{4}\r'.format(joy_data.steer, 
        #                 joy_data.throttle, joy_data.brake, velocity, end)

        # cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f} \t{4}\r'.format(pred_data.steer,
        #                                 pred_data.throttle, joy_data.brake, velocity, end)
        # sys.stdout.write(cur_output)
        # sys.stdout.flush()
            
        
        ## ready for processing a new input image
        neural_control.image_processed = False
        neural_control.rate.sleep()

def main_delta(model_path, delta_model_path):

    # ready for neural network
    neural_control = NeuralControl(model_path, delta_model_path)
    
    rospy.Subscriber(Config.data_collection['base_pose_topic'], Odometry, pos_vel_cb)
    # ready for /bolt topic publisher
    joy_pub = rospy.Publisher(Config.data_collection['vehicle_control_topic'], Control, queue_size = 10)
    joy_data = Control()
    pred_data = Control()
    delta_data = Control()

    if Config.data_collection['vehicle_name'] == 'rover':
        joy_pub4mavros = rospy.Publisher(Config.config['mavros_cmd_vel_topic'], Twist, queue_size=20)

    print('\nStart running. Vroom. Vroom. Vroooooom......')
    # print('steer \tthrt: \tbrake \tvelocity \tHz')
    print('steer \td_str \tthrt \t d_thr \tbrake \tvelocity \tHz')

    use_predicted_throttle = True if config['num_outputs'] == 2 else False
    while not rospy.is_shutdown():

        if neural_control.image_processed is False:
            continue
        
        start = time.time()
        end = time.time()
        # predicted steering angle from an input image
                
        if len(neural_control.lstm_vel) >= config['lstm_timestep'] and len(neural_control.lstm_image) >= config['lstm_timestep'] and len(neural_control.lstm_image_tb) >= config['lstm_timestep']:
            prediction, delta_prediction = neural_control.drive.run_delta((neural_control.lstm_image, neural_control.lstm_image_tb, neural_control.lstm_vel))
            # print(prediction)
            if prediction is None and delta_prediction is None:
                print("None")
            pred_data.steer     = prediction[0][0][0]
            pred_data.throttle  = prediction[1][0][0]
            pred_data.brake     = prediction[2][0][0]
            delta_data.steer    = delta_prediction[0][0][0] * float(Config.run_neural['delta_steer'])
            delta_data.throttle = delta_prediction[1][0][0] * float(Config.run_neural['delta_throttle'])
            delta_data.brake    = delta_prediction[2][0][0] * float(Config.run_neural['delta_brake'])
            
            if pred_data.throttle < 0 :
                pred_data.throttle = 0
            elif pred_data.throttle > 1 :
                pred_data.throttle = 1
            
            if pred_data.brake < 0 :
                pred_data.brake = 0
            elif pred_data.brake > 1 :
                pred_data.brake = 1
                
            if delta_data.throttle < 0 :
                delta_data.throttle = 0
            elif delta_data.throttle > 1 :
                delta_data.throttle = 1
            
            if delta_data.brake < 0 :
                delta_data.brake = 0
            elif delta_data.brake > 1 :
                delta_data.brake = 1
            
            joy_data.steer = (pred_data.steer + delta_data.steer) * Config.run_neural['scale_factor_steering']
            joy_data.throttle = (pred_data.throttle + delta_data.throttle) * Config.run_neural['scale_factor_throttle']
            joy_data.brake = pred_data.brake + delta_data.brake
        
        
        if joy_data.throttle < 0 :
            joy_data.throttle = 0
        elif joy_data.throttle > 1 :
            joy_data.throttle = 1
        
        if joy_data.brake < 0 :
            joy_data.brake = 0
        elif joy_data.brake > 1 :
            joy_data.brake = 1
            
            
        #############################
        ## very very simple controller
        ## 
        # is_sharp_turn = False
        # # if brake is not already applied and sharp turn
        # if Config.run_neural['ai_chauffeur'] is True:
        #     if neural_control.braking is False: 
        #         if velocity < Config.run_neural['velocity_0']: # too slow then no braking
        #             joy_data.throttle = Config.run_neural['throttle_default'] # apply default throttle
        #             joy_data.brake = 0
        #         elif abs(joy_data.steer) > Config.run_neural['sharp_turn_min']:
        #             is_sharp_turn = True
                
        #         if is_sharp_turn or velocity > Config.run_neural['max_vel']: 
        #             joy_data.throttle = Config.run_neural['throttle_sharp_turn']
        #             joy_data.brake = Config.run_neural['brake_val']
        #             neural_control.apply_brake()
        #         else:
        #             if use_predicted_throttle is False:
        #                 joy_data.throttle = Config.run_neural['throttle_default']
        #             joy_data.brake = 0
                    

            
        #     ##############################    
        #     ## publish mavros control topic
            
        #     if Config.data_collection['vehicle_name'] == 'rover':
        #         joy_data4mavros = Twist()
        #         if neural_control.braking is True:
        #             joy_data4mavros.linear.x = 0
        #             joy_data4mavros.linear.y = 0
        #         else: 
        #             joy_data4mavros.linear.x = joy_data.throttle*Config.run_neural['scale_factor_throttle']
        #             joy_data4mavros.linear.y = joy_data.steer*Config.run_neural['scale_factor_steering']

        #         joy_pub4mavros.publish(joy_data4mavros)

        joy_pub.publish(joy_data)

        ## print out
        # print(joy_data.steer, joy_data.throttle, joy_data.brake, velocity)
        # cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f} \t{4}\r'.format(joy_data.steer, 
        #                 joy_data.throttle, joy_data.brake, velocity, end)

        cur_output = '{0:.3f} \t{1:.9f} \t{2:.3f} \t{3:.9f} \t{4:.3f} \t{5:.3f} \t{6}\r'.format(pred_data.steer, delta_data.steer,
                                        pred_data.throttle, delta_data.throttle, joy_data.brake, velocity, end)
        sys.stdout.write(cur_output)
        sys.stdout.flush()
            
        
        ## ready for processing a new input image
        neural_control.image_processed = False
        neural_control.rate.sleep()



if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            exit('Usage:\n$ rosrun run_neural run_neural.py standard_model_name, delta_model_name')

        if config['delta_run'] is True:
            main_delta(sys.argv[1], sys.argv[2])
        else:
            main(sys.argv[1])
            
    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')