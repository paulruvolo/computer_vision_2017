#!/usr/bin/env python

"""
To run main.py, please run the following first:

roslaunch neato_node bringup.launch host:=HOST_NAME
"""
import cv2
import rospy
from cmdVelPublisher import CmdVelPublisher
from imageSubscriber import ImageSubscriber
from network import DQN

"""
CompRobo Spring 2017

This script is the main script of this project, which brings together computer vision, reinforced learning, and robot actuation.
The code works as follows:

1. An image is received and processed by ImageSubscriber
2. The processed image is used in the Network to determine the desired action based on the inputted image
3. Desired action is sent to CmdVelPublisher to actuate the robot
4. After actuation step is completed, subsequent image is sent to Network to update network parameters, thus learning.
5. This process continues indefinitely for as long as the robot is on and trying to follow the line
"""

class RobotController(CmdVelPublisher, ImageSubscriber, object):
    """
    Brain of the Robot, which inherits from ImageSubscriber and cmdVelPublisher, and initializes a network.
    Determines from ImageSubscriber how to move, which is executed in cmdVelPublisher, and these movements are
    optimized and learned through self.network
    """

    def __init__(self):
        #init ROS node
        rospy.init_node('robot_control')

        #super calls to parent classes
        super(RobotController, self).__init__()

        #initializes the work with starting parameters
        self.network = DQN(.0003, .1, .25)
        self.network.start()


    def robot_control(self, action):
        """
        Given action, will exceute a specificed behavior from the robot
        action:
         0 = forward
         1 = leftTurn
         2 = rightTurn
         3 = stop
        """
        try:
            if action < 0 or action > 3:
                raise ValueError("Action is invalid")
            self.state[action].__call__()
        except:
            # make robot stop
            print "Invalid action - stopping robot"
            self.state[3].__call__()

        self.sendMessage()
        rospy.sleep(.1)          # use desired action for 0.1 second
        self.state[3].__call__() # set robot to stop for .1 second
        self.sendMessage()
        rospy.sleep(.1)


    def run(self):
        """
        The main run loop
        """
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                #visualizes the binary image
                cv2.imshow('video_window', self.binary_image)
                cv2.waitKey(5)
                #feeds binary image into network to receive action with corresponding Q-values
                a, Q = self.network.feed_forward(self.binary_image)
                #moves based on move probable action
                self.robot_control(a[0])
                #updates the network parameters based on what happened from the action step
                self.network.update(self.binary_image)
            r.sleep()

        self.network.stop()


if __name__ == '__main__':
    #initializes Robot Controller and runs it
    node = RobotController()
    node.run()
