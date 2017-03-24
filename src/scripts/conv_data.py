#!/usr/bin/env python

from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from neato_node.msg import Bump
import rospy
from cmdVelPublisher import CmdVelPublisher
from imageSubscriber import ImageSubscriber

import cv2

import tty
import select
import sys
import termios

class Control_Robot(CmdVelPublisher, ImageSubscriber, object):

    def __init__(self):
        """ Initialize the robot control, """
        rospy.init_node('my_teleop')
        super(Control_Robot, self).__init__()

        self.state = {'i':self.forward,
                      'j':self.leftTurn,
                      'l':self.rightTurn,
                      'k':self.stopState}

        # get key interupt things
        self.settings = termios.tcgetattr(sys.stdin)
        self.key = None

        cv2.namedWindow('raw_image')

    def onKeypress(self):
        try:
            self.state[self.key].__call__()
        except:
            # on any other keypress, stop the robot
            self.state['k'].__call__()

        self.sendMessage()
        rospy.sleep(.25) # use desired action for one second
        self.state['k'].__call__() # set robot to stop
        self.sendMessage()
        rospy.sleep(.25)

    def getKey(self):
        """ Interupt (I think) that get a non interrupting keypress """
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        self.key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

    def imageView(self):
        cv2.imshow('raw_image', self.cv_image)
        cv2.waitKey(1)
        cv2.imshow('video_window', self.binary_image)
        cv2.waitKey(1)

    def writeImages(self, cv_image, binary_image, time, category):
        directory = 'test'
        directory = 'train'
        cv2.imwrite('data/{}}/binary/{}/{}.png'.format(directory, category, time), binary_image)
        cv2.imwrite('data/{}}/color/{}/{}.png'.format(directory, category, time), cv_image)

    def writeImagesMirror(self, cv_image, binary_image, time, category):
        cv2.imwrite('data/{}}/binary/{}/{}.png'.format(category, time), binary_image)
        cv2.imwrite('data/{}/color/{}/{}.png'.format(category, time), cv_image)

    def imageSave(self):
        cv_image = self.cv_image
        binary_image = self.binary_image
        time = rospy.Time.now()

        if self.key == 'i':
            self.writeImages(cv_image, binary_image, time, 'forward')
        elif self.key == 'j':
            self.writeImages(cv_image, binary_image, time, 'left')
            self.writeImagesMirror(cv_image, binary_image, time, 'right')
        elif self.key == 'l':
            self.writeImages(cv_image, binary_image, time, 'right')
            self.writeImagesMirror(cv_image, binary_image, time, 'left')

    def run(self):
        while self.key != '\x03':
            self.getKey()
            self.imageSave()
            self.onKeypress()
            self.imageView()

control = Control_Robot()
control.run()
