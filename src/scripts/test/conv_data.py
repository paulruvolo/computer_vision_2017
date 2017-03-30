#!/usr/bin/env python

"""
To run conv_data.py, please run:

roslaunch neato_node bringup.launch host:=HOST_NAME
"""

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

"""
CompRobo Spring 2017

Completed by Nathaniel Yee

This script is a test script meant to collect image data for the convNet as training.
"""

class Control_Robot(CmdVelPublisher, ImageSubscriber, object):
    """
    A class for systematically collecting data on various scenarios while following a line
    """

    def __init__(self):
        # init ROS node
        rospy.init_node('my_teleop')

        #super call to parent classes
        super(Control_Robot, self).__init__()

        #define states
        self.state = {'i':self.forward,
                      'j':self.leftTurn,
                      'l':self.rightTurn,
                      'k':self.stopState}

        # get key interupt things
        self.settings = termios.tcgetattr(sys.stdin)
        self.key = None

        # visualization purposes
        cv2.namedWindow('raw_image')


    def onKeypress(self):
        """
        moves the robot based on keypress
        """
        try:
            self.state[self.key].__call__()
        except:
            # on any other keypress, stop the robot
            self.state['k'].__call__()

        self.sendMessage()
        rospy.sleep(.25)           # use desired action for .25 second
        self.state['k'].__call__() # set robot to stop for .25 second
        self.sendMessage()
        rospy.sleep(.25)


    def getKey(self):
        """
        Interupt (I think) that get a non interrupting keypress
        """
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        self.key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


    def imageView(self):
        """
        Visualize what the robot sees
        """
        cv2.imshow('raw_image', self.cv_image)
        cv2.waitKey(1)
        cv2.imshow('video_window', self.binary_image)
        cv2.waitKey(1)


    def writeImages(self, cv_image, binary_image, time, category):
        """
        Writes images into directory for storage
        """
        # directory = 'test'
        directory = 'train'
        cv2.imwrite('data/{}/binary/{}/{}.png'.format(directory, category, time), binary_image)
        cv2.imwrite('data/{}/color/{}/{}.png'.format(directory, category, time), cv_image)


    def writeImagesMirror(self, cv_image, binary_image, time, category):
        """
        Doubles image database by mirroring each received image and storing it
        """
        # directory = 'test'
        directory = 'train'
        cv2.imwrite('data/{}/binary/{}/{}flip.png'.format(directory, category, time), cv2.flip(binary_image, 1))
        cv2.imwrite('data/{}/color/{}/{}flip.png'.format(directory, category, time), cv2.flip(cv_image, 1))


    def imageSave(self):
        """
        Stores image with appropriate labels
        """
        cv_image = self.cv_image
        binary_image = self.binary_image
        time = rospy.Time.now()

        if self.key == 'i':
            self.writeImages(cv_image, binary_image, time, 'forward')
            self.writeImagesMirror(cv_image, binary_image, time, 'forward')
        elif self.key == 'j':
            self.writeImages(cv_image, binary_image, time, 'left')
            self.writeImagesMirror(cv_image, binary_image, time, 'right')
        elif self.key == 'l':
            self.writeImages(cv_image, binary_image, time, 'right')
            self.writeImagesMirror(cv_image, binary_image, time, 'left')


    def run(self):
        """
        Main loop
        """
        while self.key != '\x03':
            # continually loops through the 4 steps to find an image, save it, move, and see it
            self.getKey()
            self.imageSave()
            self.onKeypress()
            self.imageView()


if __name__ == "__main__":
    #Initializes Control_Robot class and runs it
    control = Control_Robot()
    control.run()
