import rosbag
from cv_bridge import CvBridge
import cv2
import numpy as np

print "Ready to read rosbag"

bag = rosbag.Bag('../bags/uturn.bag')

last_cmd_vel = None
last_img = None

vel_topic = '/cmd_vel'
camera_topic = '/camera/image_raw/compressed'

bridge = CvBridge()
# print bridge.imgmsg_to_cv2

for topic, msg, t in bag.read_messages(topics=[vel_topic, camera_topic]):
    if topic == vel_topic:
        last_cmd_vel = msg
    if topic == camera_topic:
        print len(msg)
        break
        # last_img = bridge.imgmsg_to_cv2(msg)
        # last_img = cv2.resize(last_img, (32, 32))
    # print (last_cmd_vel)

bag.close()
print "Rosbag complete"
