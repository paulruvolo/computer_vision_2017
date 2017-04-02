#!/usr/bin/env python

"""
To run mask_finder.py, please run:

roslaunch neato_node bringup.launch host:=HOST_NAME
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from color_slider import Color_Slider

"""
Comprobo Spring 2017

This script is meant as a test script to find out different combinations of filtering images to find the red line
The script interacts with the color_slider class to easily discover good filters for images
The code works as follows:

1. Receives an image and processes it using current RGB filters
2. Color_Slider allows for toggling of each channel to change the filter and see results in real time
3. Manually slide colors around until desired filter is found.
"""

class MaskFinder(Color_Slider, object):
    """
    This script helps find color masks
    """

    def __init__(self):
        """
        Initialize the color mask finder
        """
        # super call to Color_Slider class
        super(MaskFinder, self).__init__()

        # init ROS node
        rospy.init_node('mask_finder')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        cv2.namedWindow('video_window')             # OpenCV window to view original image

        # ROS Subscriber to get camera feed
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)


    def process_image(self, msg):
        """
        Receives and processes a binary image based on current RGB masking parameters
        """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Binary masking using the values from the Color_Slider
        self.binary_image = cv2.inRange(self.cv_image, (self.rgb_lb[2],self.rgb_lb[1],self.rgb_lb[0]), (self.rgb_ub[2],self.rgb_ub[1],self.rgb_ub[0]))


    def run(self):
        """
        The main run loop
        """
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                # creates a window and displays the image
                cv2.imshow('video_window', self.cv_image)
                cv2.waitKey(5)
                cv2.imshow('binary_window', self.binary_image)
                cv2.waitKey(5)

            r.sleep()

if __name__ == '__main__':
    # initializes MaskFinder and runs it
    node = MaskFinder()
    node.run()
