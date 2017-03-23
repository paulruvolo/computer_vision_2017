#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import random
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class RL(object):
    def __init__(self):
        #init ROS node
        rospy.init_node('rewards')

        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV

        self.image = None                           # the latest image from the camera
        self.cv_image = None                        # gaussian blurred image
        self.binary_image = None                    # Binary form of image

        #the thresholds to find the yellow color of the sign
        self.rgb_lb = np.array([105, 0, 0]) # rgb lower bound
        self.rgb_ub = np.array([205, 75, 115]) # rgb upper bound

        # the various windows for visualization
        cv2.namedWindow('Binary_window')
        cv2.namedWindow('RGB_window')

        self.weights = np.arange(31.0, -1.0, -1.0)
        self.weights = self.weights/np.sum(self.weights)
        self.reward = 0

        #init ROS Subscriber to camera image
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)


        #RL stuff

        tf.reset_default_graph()
        #These lines establish the feed-forward part of the network used to choose actions
        self.inputs1 = tf.reshape(tf.placeholder(shape=[32,32],dtype=tf.float32), [1,1024])
        self.W = tf.Variable(tf.random_uniform([1024,3],0,0.01))
        self.Qout = tf.matmul(self.inputs1,self.W)
        self.predict = tf.argmax(self.Qout,1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[1,3],dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.updateModel = trainer.minimize(self.loss)

        self.init = tf.initialize_all_variables()

        # Set learning parameters
        self.y = .99
        self.e = 0.1
        #create lists to contain total rewards and steps per episode
        self.rList = []

        self.sess = tf.Session()
        self.sess.run(init)

    def process_image(self, msg):

        #converts ROS image to OpenCV image
        self.cv_image= self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.cv_image = cv2.resize(self.cv_image, (32, 32))
        #creates a binary image on the basis of the yellow sign
        self.binary_image = cv2.inRange(self.cv_image, (self.rgb_lb[0], self.rgb_lb[1],self.rgb_lb[2]), (self.rgb_ub[0],self.rgb_ub[1],self.rgb_ub[2]))

        self.reward = self.calculate_reward()



        #The Q-Network

        #Choose an action by greedily (with e chance of random action) from the Q-network
        a,allQ = self.sess.run([self.predict,self.Qout],feed_dict={self.inputs1:self.binary_image})
        if np.random.rand(1) < self.e:
            a[0] = np.random.random(0,3)
        #Get new state and reward from environment
        s1,r_ = env.step(a[0])
        #Obtain the Q' values by feeding the new state through our network
        Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
        #Obtain maxQ' and set our target value for chosen action.
        maxQ1 = np.max(Q1)
        targetQ = allQ
        targetQ[0,a[0]] = r + y*maxQ1
        #Train our network using target and predicted Q values
        sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
        s = s1
        #Reduce chance of random action as we train the model.
        self.e = 1./((i/50) + 10)
        rList.append(r)


    def calculate_reward(self):

        goodness = 0
        for r in range(len(self.binary_image)-1, -1, -1):
            weight = self.weights[r]
            goodness += np.sum(self.binary_image[r]*weight)

        return np.tanh(goodness)

    def run(self):
        """
        The main run loop
        """
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.image is None:
                # creates the windows and displays the RGB, HSV, and binary image
                cv2.imshow('Binary_window', self.binary_image)
                cv2.imshow('RGB_window', self.image)

                cv2.waitKey(5)
            r.sleep()



if __name__ == '__main__':
    #initializes the Sign Recognition class and runs it
    node = RL()
    node.run()
