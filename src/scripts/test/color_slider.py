#!/usr/bin/env python
"""
To run color_slider.py, please run:

roslaunch neato_node bringup.launch host:=HOST_NAME

rosrun lane_follower mask_finder.py
"""

import cv2
import numpy as np

"""
CompRobo Spring 2017

Completed by Nathaniel Yee

This script is a helper script for mask_finder.py to find optimal filtering parameters for images
It holds the color bars used to interactively change filter channels in RGB.
"""


class Color_Slider(object):
    """
    The Color_Slider class, which holds the 6 colors for RGB upper and lower bounds
    """

    def __init__(self):
        cv2.namedWindow('binary_window')
        print "Creating color sliders"

        self.rgb_lb = np.array([10,106,158]) # rgb lower bound
        cv2.createTrackbar('R lb', 'binary_window', 0, 255, self.set_r_lb)
        cv2.createTrackbar('G lb', 'binary_window', 0, 255, self.set_g_lb)
        cv2.createTrackbar('B lb', 'binary_window', 0, 255, self.set_b_lb)

        self.rgb_ub = np.array([20,255,255]) # rgb upper bound
        cv2.createTrackbar('R ub', 'binary_window', 0, 255, self.set_r_ub)
        cv2.createTrackbar('G ub', 'binary_window', 0, 255, self.set_g_ub)
        cv2.createTrackbar('B ub', 'binary_window', 0, 255, self.set_b_ub)


    def set_r_lb(self, val):
        """
        set red lower bound
        """
        self.rgb_lb[0] = val


    def set_g_lb(self, val):
        """
        set green lower bound
        """
        self.rgb_lb[1] = val


    def set_b_lb(self, val):
        """
        set blue lower bound
        """
        self.rgb_lb[2] = val


    def set_r_ub(self, val):
        """
        set red upper bound
        """
        self.rgb_ub[0] = val


    def set_g_ub(self, val):
        """
        set green upper bound
        """
        self.rgb_ub[1] = val


    def set_b_ub(self, val):
        """
        set blue upper bound
        """
        self.rgb_ub[2] = val
