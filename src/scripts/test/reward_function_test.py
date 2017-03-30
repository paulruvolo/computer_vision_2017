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

This script is meant as a test file to try out different reward/loss functions in a lightweight method rather than running our entire network.
The code performs very basic functionality:

1. Receives and processes an image into a binary output
2. Runs and prints the output of the reward function
"""

class Reward(object):
    """
    The Reward Object is a test object used to try out different versions of the reward function
    """

    def __init__(self):
        #init ROS node
        rospy.init_node('rewards')

        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV

        self.image = None                           # the latest image from the camera
        self.cv_image = None                        # resized image
        self.binary_image = None                    # Binary form of image

        #the thresholds to find the red color of the line
        self.rgb_lb = np.array([105, 0, 0]) # hsv lower bound
        self.rgb_ub = np.array([205, 90, 60]) # hsv upper bound

        # the various windows for visualization
        cv2.namedWindow('Binary_window')
        cv2.namedWindow('RGB_window')

        self.reward = 0                                              # the reward variable
        self.reward_window = (np.arange(20,32), np.arange(0,32))     # reward only looks at bottom part of image
        self.previous_reward = 0                                     # the previous reward to calculate new reward

        #init ROS Subscriber to camera image
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)


    def process_image(self, msg):
        """
        Callback that takes a ROS image and converts it into a binary image for use in the reward function
        """

        #converts ROS image to OpenCV image
        self.image= self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        #blurs image to average out some color
        self.cv_image = cv2.resize(self.image, (32, 32))
        #creates a binary image on the basis of the yellow sign
        self.binary_image = cv2.inRange(self.cv_image, (self.rgb_lb[2], self.rgb_lb[1],self.rgb_lb[0]), (self.rgb_ub[2],self.rgb_ub[1],self.rgb_ub[0]))

        self.calculate_reward()

        print self.reward


    def calculate_reward(self):
        """
        reward function, currently using the latest version of the reward function
        """

        goodness = 0
        self.reward = 0
        for r in self.reward_window[0]:
            for c in self.reward_window[1]:
                goodness += (binary[r][c] == 1.0)
        print goodness, self.previous_reward
        if goodness > self.previous_reward:
            self.reward =  1
        elif (goodness < self.previous_reward) or (goodness == self.previous_reward and goodness == 0):
            self.reward = -1
        else:
            self.reward = 0
        self.previous_reward = goodness


    def run(self):
        """
        The main run loop
        """
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.image is None:
                # creates the windows and displays the RGB, and binary image
                cv2.imshow('Binary_window', self.binary_image)
                cv2.imshow('RGB_window', self.image)

                cv2.waitKey(5)
            r.sleep()


if __name__ == '__main__':
    #initializes the Reward class and runs it
    node = Reward()
    node.run()
