#!/usr/bin/env python
"""
To run network.py, please run:

roslaunch neato_node bringup.launch host:=HOST_NAME

rosrun lane_follower main.py
"""

import numpy as np
import random
import tensorflow as tf

"""
CompRobo Spring 2017

This script holds the learning engine our project. It consists of a two-layer convNet with a Q-learning algorithm.
The learning works as follows:

1. Receive a pre-processed image as input.
2. Image goes through two layers of the convNet to extract main features and condense information
3. Then it goes into Q-learning as input, which will output Q-values for the three ways it can move: left, right, or forward
4. It then selects a movement and exceutes it
5. Finally, it updates that movement's Q-value based on the reward obtained from the movemenet's image input.
6. It continues this indefinitely, until the Q-values for each movement reach optimal values and converge
"""

class DQN(object):
    """
    The Q-Learning Reinforced Learning Class, which holds our network that learns how to follow a line
    """

    def __init__(self, lr, y, e):
        # hyper parameters
        self.lr = lr    # learning rate
        self.y = y      # constant applied to Q-value when it's learning after an action is performed to update Q-value
        self.e = e      # random action probablity
        self.i = 0      # rate of random action probability decay over time

        # These lines establish the feed-forward part of the network

        # used to choose actions
        self.input = tf.placeholder(shape=[32,32],dtype=tf.float32)
        self.X = tf.reshape(self.input, [-1,32,32,1])

        # first conv layer
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        #max-pool intermediate layer
        h_conv1 = tf.nn.relu(conv2d(self.X, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # second conv layer
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        # second max-pool intermediate layer
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # fully-connected 1st layer
        W_fc1 = weight_variable([8 * 8 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # fully-connected 2nd layer
        W_fc2 = weight_variable([1024, 3])
        b_fc2 = bias_variable([3])
        self.output = tf.matmul(h_fc1, W_fc2) + b_fc2
        self.predict = tf.argmax(self.output, 1)

        # Below we obtain the loss by taking the sum of squares
        # difference between the target and prediction Q values.
        self.target = tf.placeholder(shape=[1,3],dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.target - self.output))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate = self.lr)
        self.updateModel = self.trainer.minimize(self.loss)

        # initialize session
        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()

        # action set: 0 is forward, 1 is left, 2 is right, and 3 is stop
        self.actions = [0, 1, 2]

        # reward variables
        self.reward = 0                                              # the reward variable
        self.reward_window = (np.arange(20,32), np.arange(0,32))     # reward only looks at bottom part of image
        self.previous_goodness = 0                                   # the previous loss used to calculate new reward


    def start(self):
        """
        start a session
        """
        self.sess.run(self.init)


    def stop(self):
        """
        end a session
        """
        self.sess.close()


    def update(self, state):
        """
        compute loss and update the network with next state and reward
        """
        #calculate reward
        self.reward = self.calculate_reward(state)

        # obtain the Q' values by feeding the new state through the network
        Q1 = self.sess.run(self.output,feed_dict={self.input:state})
        max_Q1 = np.max(Q1)
        target_Q = self.Q
        target_Q[0, self.a[0]] = self.reward + self.y * max_Q1

        # train our network using target and predicted Q values
        self.sess.run([self.updateModel],
            feed_dict={self.input:self.current_state,self.target:target_Q})


    def feed_forward(self, state):
        """
        feed forward the network with a state to get an action vector
        """
        # Choose an action by greedily (with e chance of random action) from the Q-network
        self.a, self.Q = self.sess.run([self.predict, self.output],
            feed_dict={self.input:state})

        self.current_state = state

        # e chance to select a random action
        if np.random.rand(1) < self.e:
            self.a[0] = self.get_random_action()

        #decay the randomization factor
        self.i += 1
        self.e = 1./((self.i/100.0) + 10)

        return self.a, self.Q


    def calculate_reward(self, binary):
        """
        calculates the reward from the image after action is taken
        """
        goodness = 0  # amount of white pixels in processed image
        reward = 0    # current reward

        # iterate through reward_window and find the cumulative goodness
        for r in self.reward_window[0]:
            for c in self.reward_window[1]:
                goodness += (binary[r][c] == 1.0)

        # calculate reward based on comparison of previous goodness with current goodness
        if goodness > self.previous_goodness:
            reward =  1
        elif (goodness < self.previous_goodness) or (goodness == self.previous_goodness and goodness == 0):
            reward = -1
        else:
            reward = 0
        self.previous_goodness = goodness

        return reward


    def get_random_action(self):
        """
        get a random action from actions
        """
        return random.choice(self.actions)


######################### helper functions for convNet #################################

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')
