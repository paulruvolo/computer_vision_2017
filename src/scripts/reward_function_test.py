#!/usr/bin/env python
"""
To run reward_function_test.py, please run:

roslaunch neato_node bringup.launch host:=HOST_NAME
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

"""
CompRobo Spring 2017

Computer Vision Project

Completed by Kevin Zhang
"""

class Reward(object):

    def __init__(self):
        #init ROS node
        rospy.init_node('rewards')

        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV

        self.image = None                           # the latest image from the camera
        self.cv_image = None                        # gaussian blurred image
        self.hsv_image = None                       # HSV form of image
        self.binary_image = None                    # Binary form of image

        #the thresholds to find the yellow color of the sign
        self.rgb_lb = np.array([105, 0, 0]) # hsv lower bound
        self.rgb_ub = np.array([205, 75, 115]) # hsv upper bound

        # the various windows for visualization
        cv2.namedWindow('HSV_window')
        cv2.namedWindow('Binary_window')
        cv2.namedWindow('RGB_window')

        self.weights = np.arange(479.0, -1.0, -1.0)
        self.weights = self.weights/np.sum(self.weights)
        self.reward = 0
        self.proxy_decision = None

        #init ROS Subscriber to camera image
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)


    def process_image(self, msg):

        #converts ROS image to OpenCV image
        self.image= self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        #blurs image to average out some color
        self.cv_image = cv2.medianBlur(self.image,5)
        #converts RGB image to hue, saturation, value image
        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        #creates a binary image on the basis of the yellow sign
        self.binary_image = cv2.inRange(self.cv_image, (self.rgb_lb[0], self.rgb_lb[1],self.rgb_lb[2]), (self.rgb_ub[0],self.rgb_ub[1],self.rgb_ub[2]))

        self.reward = self.calculate_reward()

        print self.reward

    def calculate_reward(self):

        cost = 0
        for r in range(len(self.binary_image)-1, -1, -1):
            weight = self.weights[r]
            cost += np.sum(self.binary_image[r]*weight)

        return cost


    def run(self):
        """
        The main run loop
        """
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.image is None:
                # creates the windows and displays the RGB, HSV, and binary image
                cv2.imshow('HSV_window', self.hsv_image)
                cv2.imshow('Binary_window', self.binary_image)
                cv2.imshow('RGB_window', self.image)

                cv2.waitKey(5)
            r.sleep()



if __name__ == '__main__':
    #initializes the Sign Recognition class and runs it
    node = Reward()
    node.run()
