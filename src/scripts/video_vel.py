#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist

class MaskFinder(object):
    """ This script helps find colors masks """


    def __init__(self):
        rospy.init_node('get_image')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        rospy.Subscriber("/cmd_vel", Twist ,self.process_cmd_vel)

        cv2.namedWindow('video_window')
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.cv_image = cv2.resize(self.cv_image, (32, 32))

    def process_cmd_vel(self, msg):
        print "new msg"
        print msg



    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                cv2.imshow('video_window', self.cv_image)
                cv2.waitKey(5)
            r.sleep()

if __name__ == '__main__':
    node = MaskFinder()
    node.run()
