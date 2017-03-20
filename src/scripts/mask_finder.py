#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from color_slider import Color_Slider

class MaskFinder(Color_Slider, object):
    """ This script helps find colors masks """


    def __init__(self):
        """ Initialize the color mask finder """


        super(MaskFinder, self).__init__()
        # initialize the template matcher

        rospy.init_node('mask_finder')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        cv2.namedWindow('video_window')
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)

        # uncomment for viewing hsv values in image
        self.image_info_window = None
        cv2.setMouseCallback('video_window', self.process_mouse_event)


    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)

        # use user specified colors here
        # self.binary_image = cv2.inRange(self.hsv_image, (10,106,158), (20,255,255))

        # uncomment for color sliders
        self.binary_image = cv2.inRange(self.cv_image, (self.rgb_lb[2],self.rgb_lb[1],self.rgb_lb[0]), (self.rgb_ub[2],self.rgb_ub[1],self.rgb_ub[0]))

    def process_mouse_event(self, event, x,y,flags,param):
        """ Process mouse events so that you can see the color values associated
        with a particular pixel in the camera images """
        self.image_info_window = 255*np.ones((500,500,3))

        # show hsv values
        cv2.putText(self.image_info_window,
        'Color (r=%d,g=%d,b=%d)' % (self.cv_image[y,x,2], self.cv_image[y,x,1], self.cv_image[y,x,0]),
        (5,50), # 5 = x, 50 = y
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,0))

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                # print "here"
                # creates a window and displays the image for X milliseconds
                cv2.imshow('video_window', self.cv_image)
                cv2.waitKey(5)
                cv2.imshow('binary_window', self.binary_image)
                cv2.waitKey(5)
            if not self.image_info_window is None:
                cv2.imshow('image_info', self.image_info_window)
                cv2.waitKey(5)
            r.sleep()

if __name__ == '__main__':
    node = MaskFinder()
    node.run()
