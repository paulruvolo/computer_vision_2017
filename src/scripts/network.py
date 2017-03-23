import numpy as np
import random
import tensorflow as tf

class DQN(object):

    def __init__(self, lr, y, e):
        # hyper parameters
        self.lr = lr    # learning rate
        self.y = y
        self.e = e      # random action probablity
        self.i = 0
        # These lines establish the feed-forward part of the network
        # used to choose actions
        self.input = tf.placeholder(shape=[1,1024],dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([1024,3],0,0.01))
        self.output = tf.matmul(self.input, self.W)
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

        # action set
        self.actions = [0, 1, 2] # TODO: change to enum

        self.reward = 0
        self.reward_window = (np.arange(28,33), np.arange(14,20))
        self.indices = self.calculate_index()

    def start(self):
        """start a session"""
        self.sess.run(self.init)

    def stop(self):
        """end a session"""
        self.sess.close()

    def update(self, state):
        """ compute loss and update the network with next state and reward"""

        self.reward = self.calculate_reward(state)
        # obtain the Q' values by feeding the new state through the network

        Q1 = self.sess.run(self.output,feed_dict={self.input:state})
        max_Q1 = np.max(Q1)
        target_Q = self.Q
        target_Q[0, self.a[0]] = self.reward + self.y * max_Q1

        # train our network using target and predicted Q values
        self.sess.run([self.updateModel,self.W],
            feed_dict={self.input:self.current_state,self.target:target_Q})

    def feed_forward(self, state):
        """feed forward the network with a state to get an action vector"""

        # Choose an action by greedily (with e chance of random action) from the Q-network
        self.a, self.Q = self.sess.run([self.predict, self.output],
            feed_dict={self.input:state})
        self.current_state = state

        # e chance to select a random action
        if np.random.rand(1) < self.e:
            self.a[0] = self.get_random_action()

        self.i += 1
        # self.e = 1./((self.i/50.0) + 10)

        return self.a, self.Q

    def calculate_reward(self, binary):
        """calculates the reward from the image"""

        goodness = 0

        for index in self.indices:
            goodness += (binary[0][index] == 255)

        if goodness > 10:
            return 1
        else:
            return 0
    def calculate_index(self):
        """calculate index for reward calculation"""
        indices = []
        for r in self.reward_window[0]:
            for c in self.reward_window[1]:
                indices.append((r-1)*32 + c-1)
        return indices
    def get_random_action(self):
        """get a random action from actions"""
        return random.choice(self.actions)
