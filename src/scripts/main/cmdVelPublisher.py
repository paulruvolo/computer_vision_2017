#!/usr/bin/env python
"""
To run cmdVelPublisher.py, please run:

roslaunch neato_node bringup.launch host:=HOST_NAME

rosrun lane_follower main.py
"""

from geometry_msgs.msg import Twist, Vector3
import rospy

"""
CompRobo Spring 2017

This script is a helper script for the RobotController class, and executes actions decided on by the network
It basically creates Twist message and then specifies the message based on the inputted action
"""

class CmdVelPublisher(object):
    """
    The helper class for actuation for RobotController
    """

    def __init__(self):
        #super call because RobotController performs two super calls
        super(CmdVelPublisher, self).__init__()

        # ROS publisher to cmd_vel, the topic the Neato listens to for movements
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # initializes two vectors, one for turning, one for straight movement
        self.linearVector = Vector3(x=0.0, y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=0.0)

        # defines the states
        self.state = {0:self.forward,
                      1:self.leftTurn,
                      2:self.rightTurn,
                      3:self.stopState}
        print "Initialized CmdVelPublisher"


    def forward(self):
        """
        Sets the velocity to forward
        """
        print('forward')
        self.linearVector  = Vector3(x=0.1, y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=0.0)


    def leftTurn(self):
        """
        Sets the velocity to turn left
        """
        print('leftTurn')
        self.linearVector  = Vector3(x=0.1, y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=0.7)


    def rightTurn(self):
        """
        Sets the velocity to turn right
        """
        print('rightTurn')
        self.linearVector  = Vector3(x=0.1, y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=-0.7)


    def stopState(self):
        """
        Sets the velocity to stop
        """
        self.linearVector  = Vector3(x=0.0, y=0.0, z=0.0)
        self.angularVector = Vector3(x=0.0, y=0.0, z=0.0)


    def sendMessage(self):
        """
        Publishes the Twist containing the linear and angular vector
        """
        self.pub.publish(Twist(linear=self.linearVector, angular=self.angularVector))
