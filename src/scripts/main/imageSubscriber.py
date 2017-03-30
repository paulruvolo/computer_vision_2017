#!/usr/bin/env python
"""
To run imageSubscriber.py, please run:

roslaunch neato_node bringup.launch host:=HOST_NAME

rosrun lane_follower main.py
"""

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy
import numpy as np

"""
CompRobo Spring 2017

This script is a helper class for the RobotController class, which pre-processes received images for the network.
Pre-processing is defined as creating a binary image based on a specific shade of red, which is the color of the line we are following.
"""

class ImageSubscriber(object):
    """
    The Image processing helper class, which processes an image all within a well-made package for RobotController
    """

    def __init__(self):
        #super call because of multiple super calls in RobotController
        super(ImageSubscriber, self).__init__()

        self.cv_image = None                        # the latest image from the camera
        self.binary_image = None                    # the processed binary image

        #the thresholds to find the red color of the line
        self.rgb_lb = np.array([105, 0, 0]) # rgb lower bound
        self.rgb_ub = np.array([205, 90, 60]) # rgb upper bound

        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        cv2.namedWindow('video_window')             # the OpenCV window to visualize the image

        # ROS subscriber to camera feed
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)
        print "Initialize ImageSubscriber"


    def process_image(self, msg):
        """
        Receives an image and masks it with the red filter to create a binary image for use in the network
        """
        self.cv_image = cv2.resize(self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"), (32, 32))

        #create the binary mask, and also normalizes the values in the image, so everything in the matrix is 1 or 0
        self.binary_image = cv2.inRange(self.cv_image, (self.rgb_lb[2], self.rgb_lb[1],self.rgb_lb[0]), (self.rgb_ub[2],self.rgb_ub[1],self.rgb_ub[0]))/255.0
