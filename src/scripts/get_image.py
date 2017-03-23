#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist, Vector3

class MaskFinder(object):
    """ This script helps find colors masks """


    def __init__(self):
        rospy.init_node('get_image')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        cv2.namedWindow('video_window')
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.cv_image = cv2.resize(self.cv_image, (32, 32))

    def leftTurn(self):
        print('leftTurn')
        """ Sets the velocity to turn left """
        self.linearVector  = Vector3(x=0.0, y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=1.0)

    def rightTurn(self):
        print('rightTurn')
        """ Sets the velocity to turn right """
        self.linearVector  = Vector3(x=0.0, y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=-1.0)

    def sendMessage(self):
        """ Publishes the Twist containing the linear and angular vector """
        print('sendMessage')
        self.pub.publish(Twist(linear=self.linearVector, angular=self.angularVector))

    def robot_control(self, action):
        # action:
        # 0 = forward
        # 1 = leftTurn
        # 2 = rightTurn
        # 3 = stop
        try:
            if action < 0 or action > 3:
                raise ValueError("Action is invalid")
            self.state[action].__call__()
        except:
            # make robot stop
            print "Invalid action - stopping robot"
            self.state[3].__call__()

        self.sendMessage()
        rospy.sleep(1) # use desired action for one second
        self.state[3].__call__() # set robot to stop
        self.sendMessage()
        rospy.sleep(.25)


    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                cv2.imshow('video_window', self.cv_image)
                cv2.waitKey(5)
                self.robot_control(3)
            r.sleep()

if __name__ == '__main__':
    node = MaskFinder()
    node.run()
