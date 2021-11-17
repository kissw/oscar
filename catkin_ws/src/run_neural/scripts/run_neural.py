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
        self.vel = None
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
        self.image = crop_str
        self.image_crop = crop_tb
        self.vel = velocity
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

def fusion_cb(value):
    global steering
    steering = value.steer

def main(model_path):
    # ready for neural network
    neural_control = NeuralControl(model_path, model_path)
    
    rospy.Subscriber(Config.data_collection['base_pose_topic'], Odometry, pos_vel_cb)
    # ready for /bolt topic publisher
    joy_pub = rospy.Publisher(Config.data_collection['vehicle_control_topic'], Control, queue_size = 10)
    delta_pub = rospy.Publisher(Config.data_collection['vehicle_control_delta_topic'], Control, queue_size = 10)
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
        
        if config['lstm'] == True: 
            if len(neural_control.lstm_vel) >= config['lstm_timestep']:
                # prediction = neural_control.drive.run((neural_control.image, neural_control.image_crop, neural_control.vel))
                prediction = neural_control.drive.run((neural_control.lstm_image, neural_control.lstm_image_tb, neural_control.lstm_vel))
                
                pred_data.steer     = prediction[0][0][0]
                pred_data.throttle  = prediction[1][0][0]
                
                if pred_data.throttle < 0 :
                    pred_data.throttle = 0
                elif pred_data.throttle > 1 :
                    pred_data.throttle = 1
                
                joy_data.steer = (pred_data.steer) * Config.run_neural['scale_factor_steering']
                joy_data.throttle = (pred_data.throttle) * Config.run_neural['scale_factor_throttle']
        else:
            prediction = neural_control.drive.run((neural_control.image, neural_control.image_crop, neural_control.vel))
            # prediction = neural_control.drive.run((neural_control.lstm_image, neural_control.lstm_image_tb, neural_control.lstm_vel))
            
            pred_data.steer     = prediction[0][0][0]
            pred_data.throttle  = prediction[1][0][0]
            # print(pred_data)
            if pred_data.throttle < 0 :
                pred_data.throttle = 0
            elif pred_data.throttle > 1 :
                pred_data.throttle = 1
            
            joy_data.steer = (pred_data.steer) * Config.run_neural['scale_factor_steering']
            joy_data.throttle = (pred_data.throttle) * Config.run_neural['scale_factor_throttle']
        if joy_data.throttle < 0 :
            joy_data.throttle = 0
        elif joy_data.throttle > 1 :
            joy_data.throttle = 1
        
        joy_pub.publish(joy_data)
        # delta_pub.publish(delta_data_raw)

        cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f} \t{4}\r'.format(pred_data.steer,
                                        pred_data.throttle, joy_data.brake, velocity, end)
        sys.stdout.write(cur_output)
        sys.stdout.flush()
            
        
        ## ready for processing a new input image
        neural_control.image_processed = False
        neural_control.rate.sleep()

def main_delta(model_path, delta_model_path):

    # ready for neural network
    neural_control = NeuralControl(model_path, delta_model_path)
    
    rospy.Subscriber('/fusion2', Control, fusion_cb)
    rospy.Subscriber(Config.data_collection['base_pose_topic'], Odometry, pos_vel_cb)
    # ready for /bolt topic publisher
    joy_pub = rospy.Publisher(Config.data_collection['vehicle_control_topic'], Control, queue_size = 10)
    delta_pub = rospy.Publisher(Config.data_collection['vehicle_control_delta_topic'], Control, queue_size = 10)
    joy_data = Control()
    pred_data = Control()
    delta_data = Control()
    delta_data_raw = Control()

    if Config.data_collection['vehicle_name'] == 'rover':
        joy_pub4mavros = rospy.Publisher(Config.config['mavros_cmd_vel_topic'], Twist, queue_size=20)

    print('\nStart running. Vroom. Vroom. Vroooooom......')
    # print('steer \tthrt: \tbrake \tvelocity \tHz')
    print('steer \td_str \tthrt \td_thr \tbrake \tvelocity \tHz')

    use_predicted_throttle = True if config['num_outputs'] == 2 else False
    while not rospy.is_shutdown():

        if neural_control.image_processed is False:
            continue
        
        start = time.time()
        # predicted steering angle from an input image
                
        if len(neural_control.lstm_vel) >= config['lstm_timestep'] and len(neural_control.lstm_image) >= config['lstm_timestep'] and len(neural_control.lstm_image_tb) >= config['lstm_timestep']:
            prediction, delta_prediction = neural_control.drive.run_delta((neural_control.image, neural_control.image_crop, neural_control.vel,neural_control.lstm_image, neural_control.lstm_image_tb, neural_control.lstm_vel))
            if prediction is None and delta_prediction is None:
                print("None")
            pred_data.steer     = prediction[0][0][0]
            pred_data.throttle  = prediction[1][0][0]
            # print(pred_data.steer)
            # print(delta_prediction)
            # pred_data.brake     = prediction[2][0][0]
            # delta_data_raw.steer    = delta_prediction[0][0]
            delta_data_raw.throttle = delta_prediction[0][0]
            # delta_data_raw.brake    = delta_prediction[2][0][0]
            # delta_data.steer    = delta_data_raw.steer    * float(Config.run_neural['delta_steer'])
            # delta_data.steer    = 0
            delta_data.throttle = delta_data_raw.throttle * float(Config.run_neural['delta_throttle'])
            # delta_data.brake    = delta_data_raw.brake    * float(Config.run_neural['delta_brake'])
            # delta_data.brake    = 0
            
            if pred_data.throttle < 0 :
                pred_data.throttle = 0
            elif pred_data.throttle > 1 :
                pred_data.throttle = 1
            if delta_data.throttle < 0 :
                delta_data.throttle = 0
            elif delta_data.throttle > 1 :
                delta_data.throttle = 1
                
            joy_data.steer = steering
            joy_data.throttle = (pred_data.throttle + delta_data.throttle) * Config.run_neural['scale_factor_throttle']
        
        
        if joy_data.throttle < 0 :
            joy_data.throttle = 0
        elif joy_data.throttle > 1 :
            joy_data.throttle = 1
        
        if joy_data.brake < 0 :
            joy_data.brake = 0
        elif joy_data.brake > 1 :
            joy_data.brake = 1
            
        joy_pub.publish(joy_data)
        delta_pub.publish(delta_data_raw)
        
        cur_output = '{0:.3f} \t{1:.3f} \t{2:.9f} \t{3:.3f} \t{4:.3f} \t{5}\r'.format(pred_data.steer, delta_data.steer*100,
                                        pred_data.throttle, delta_data.throttle*100, velocity, 1/abs(time.time()-start))
        # cur_output = '{0:.3f} \t{1:.9f} \r'.format(pred_data.steer, delta_data.steer)
        
        sys.stdout.write(cur_output)
        sys.stdout.flush()
            
        
        ## ready for processing a new input image
        neural_control.image_processed = False
        neural_control.rate.sleep()



if __name__ == "__main__":
    try:

        if config['delta_run'] is True:
            if len(sys.argv) != 3:
                exit('Usage:\n$ rosrun run_neural run_neural.py standard_model_name, delta_model_name')
            main_delta(sys.argv[1], sys.argv[2])
        else:
            if len(sys.argv) != 2:
                exit('Usage:\n$ rosrun run_neural run_neural.py model_name')
            main(sys.argv[1])
            
    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')