import cv2
import numpy as np


class Color_Slider(object):

    def __init__(self):
        cv2.namedWindow('threshold_image')
        print "Creating color sliders"
        self.rgb_lb = np.array([10,106,158]) # hsv lower bound
        cv2.createTrackbar('R lb', 'threshold_image', 0, 255, self.set_r_lb)
        cv2.createTrackbar('G lb', 'threshold_image', 0, 255, self.set_g_lb)
        cv2.createTrackbar('B lb', 'threshold_image', 0, 255, self.set_b_lb)
        self.rgb_ub = np.array([20,255,255]) # hsv upper bound
        cv2.createTrackbar('R ub', 'threshold_image', 0, 255, self.set_r_ub)
        cv2.createTrackbar('G ub', 'threshold_image', 0, 255, self.set_g_ub)
        cv2.createTrackbar('B ub', 'threshold_image', 0, 255, self.set_b_ub)

    def set_r_lb(self, val):
        """ set red lower bound """
        self.rgb_lb[0] = val

    def set_g_lb(self, val):
        """ set green lower bound """
        self.rgb_lb[1] = val

    def set_b_lb(self, val):
        """ set blue lower bound """
        self.rgb_lb[2] = val

    def set_r_ub(self, val):
        """ set red upper bound """
        self.rgb_ub[0] = val

    def set_g_ub(self, val):
        """ set green upper bound """
        self.rgb_ub[1] = val

    def set_b_ub(self, val):
        """ set blue upper bound """
        self.rgb_ub[2] = val
