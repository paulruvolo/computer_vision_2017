import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy
import numpy as np

class ImageSubscriber(object):
    def __init__(self):
        super(ImageSubscriber, self).__init__()
        self.cv_image = None                        # the latest image from the camera
        self.binary_image = None

        #the thresholds to find the yellow color of the sign
        self.rgb_lb = np.array([105, 0, 0]) # rgb lower bound
        self.rgb_ub = np.array([205, 90, 60]) # rgb upper bound

        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        cv2.namedWindow('video_window')
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)
        print "Initialize ImageSubscriber"

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
        called cv_image for subsequent processing """
        self.cv_image = cv2.resize(self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"), (32, 32))
        self.binary_image = cv2.inRange(self.cv_image, (self.rgb_lb[2], self.rgb_lb[1],self.rgb_lb[0]), (self.rgb_ub[2],self.rgb_ub[1],self.rgb_ub[0]))/255.0
        self.binary_reshaped = self.binary_image.reshape([1,1024])
