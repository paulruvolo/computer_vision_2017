#!/usr/bin/env python

import cv2
import rospy
from cmdVelPublisher import CmdVelPublisher
from imageSubscriber import ImageSubscriber
from network import DQN

class RobotController(CmdVelPublisher, ImageSubscriber, object):
    """ This script helps find colors masks """

    def __init__(self):
        rospy.init_node('robot_control')
        super(RobotController, self).__init__()

        self.network = DQN(.001, .1, .25)
        self.network.start()


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
        rospy.sleep(.1) # use desired action for one second
        self.state[3].__call__() # set robot to stop
        self.sendMessage()
        rospy.sleep(.25)


    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                cv2.imshow('video_window', self.binary_image)
                cv2.waitKey(5)
                a, Q = self.network.feed_forward(self.binary_image)
                self.robot_control(a[0])
                self.network.update(self.binary_image)
                print self.network.reward
            r.sleep()

        self.network.stop()

if __name__ == '__main__':
    node = RobotController()
    node.run()
