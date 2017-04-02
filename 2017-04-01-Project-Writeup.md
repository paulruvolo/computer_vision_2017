---
title: Project Writeup
layout: post
author: hooticambo
permalink: /project-writeup/
source-id: 10pGuWpS3OHzSSB7CCsJy4qz2qgNLtecjF7tmhxA3XSs
published: true
---
# Autonomous Q-learning Line Following Robot

[Yuzhong Huang](https://github.com/YuzhongHuang), [Nathan Yee](https://github.com/NathanYee), [Kevin Zhang](https://github.com/kzhang8850)

## ![image alt text]({{ site.url }}/public/OsRMiLcfOSn13tNMyGPtw_img_0.jpg)

## Project Goal

Our project demonstrates that a robot can learn to follow lines in real time with a combination of computer vision, Q-learning, and convolutional neural networks.

## How it's done.

The main engine of the project consists of a two layer convolutional neural network attached to a Q-learning (Reinforced Learning) algorithm. There are three main components to the system: the Image Processor, the Network, and the Actuator. The Image Processor takes in images from the Neato's camera, resizes them into a 32x32 image and then converts them into a binary image that filters specifically for the red tape line. Each image is then used as an input to the Neural Network, where it passes through the 2 layers of the convolutional layers and then through the Q-Learning algorithm to generate probability values for each possible action. The Network will then output the most probable action based on the probabilities to the Actuator, which will then move Neato. After the Neato has moved, the Image Processor takes a new picture that  is put back into the Network to update the probability values of the previous step. In this manner the Network "learns", and over many iterations the probability space becomes optimized to follow the line.

## Design Decisions

One decision we made was to use Tensorflow over Keras as our primary Learning package. Keras wraps around Tensorflow and makes things nice to work with, but we wanted to learn more about Reinforced Learning and how it works, so we decided to go with TensorFlow because it's more low-level and offers more insight into how Q-Learning and layers function together.

Another decision was to make a two layer Convolutional Neural Network over a single layer Neural Network. The two layer ConvNet is more complicated, but for the purposes of image processing, it should converge much faster and lead to more accurate results. Our original single fully connected layer first was not able to learn fast enough. The additional two layer convolutional layers are just sophisticated enough such that our learning was tangible and accurate.

We also made some decisions on the reward function. Originally we decided to look at the entire binary image and count the number of white pixels in the 640x480 image - linearly weighting the bottom pixels to be more than the top pixels, but this method didn't allow for negative reinforcement. Next, we tried simplifying the image and went for a function that just looked at the bottom center 30 pixels of a 32x32 image and counted how many were white, with a threshold determining reward and punishment. However, this was too rigid and difficult to properly tune. We ended up using a function that looked at the bottom half of the 32x32 image and counted the number of white pixels, then it compared that with the previous reward output to determine the reward or punishment depending on whether it was higher or lower than the previously recorded value. There are still some loopholes, but for our purposes this was accurate enough.

Another design decision we made is to take out the dropout layer in the convolutional neural network since dropout layers make the network unstable. One possible explanation is that in supervised learning, mini-batches increase the complexity of data, and adding noise can help reduce overfitting without adding too much instability to the network. Reinforcement Learning back-propagates only a single state data each iteration, so adding noise without much complexity will make the network unstable. 

For the input of the neural network we chose to convert our original image from an RGB ![image alt text]({{ site.url }}/public/OsRMiLcfOSn13tNMyGPtw_img_1.png)640x480 pixel image to a binary 32x32 image. This ultimately allows us to evaluate and train our neural network faster and reduce the complexity of the network. 

## Code Structure

4 Classes:

1. RobotController (Main): Center module, combined other classes together into a cohesive structure

2. DQN: Neural Network/Q-Learning module, held the network that performed Reinforced Learning behavior

3. ImageSubscriber: Pre-processing module, received an image and sent a binary version to the main

4. CmdVelPublisher: Execution module, received commands that it sent to the Neato for movement

Code Architecture: 

Class RobotController:

* Subscribers

    * /camera/image_raw - inherit from ImageSubscriber Class

* Publishers

    * /cmd_vel - inherit from CmdVelPublisher Class

* Attributes

    * Instance of DQN (Deep Q Learning Network) Class

RobotControl control loop:

![image alt text]({{ site.url }}/public/OsRMiLcfOSn13tNMyGPtw_img_2.png)

## Challenges

Reward functions make or break reinforcement learning. A bad reward function makes it impossible for the network to learn the proper behavior. A good reward function needs to encourage good behavior without being explicit. It took us several iterations to create a sufficient reward function that allowed for proper learning.  We choose to score images by counting the number of white pixels in the image before and after the chosen action. If the before image had a higher score than the after image, we gave a score of -1 to the network. If the first image had the same score as the after image, we gave a score of 0 to the network. If the before image has a higher score than the after image we have a score of +1 to the network. Given this very generic non problem specific reward function, our network was able learn to follow the line. See design decisions for the other reward function we tried.

Learning rate explosion can be caught early by printing the output vector during the control loop. There was a time when we realized the Neato kept losing its focus on the line after a short period every run. After many attempts to debug, we finally printed our Q-values and found that they had skyrocketed exponentially in less than 2 seconds. This helped determine that the learning rate was too high when the output vector displays [nan, nan, nan] after three training cycles.

## Future Work

Right now our robot only takes discrete states and outputs discrete actions. In the future, we hope to implement policy gradient or normalized advantage functions to enable continuous learning.

Further optimizing our Q-learning algorithm could also be a thing, as right now it moves pretty slowly or sometimes gets confused, which might be due to some overfitting or lack of proper parameter tuning.

## Lessons Learned

Reinforcement Learning, or machine learning in general, is just a bunch of vectors being operated on. The "learning" is really just tweaking the vectors and “Q-value” probabilities that determine where to move and change based on what it sees after it moves.

A good reward function is critical to the successful convergence of the neural network.

Preprocessing input images can lead to faster convergence of the neural network.

There are numerous ways to implement a Neural Network or optimize it, and whether a particular way works better for you depends highly on the input you're giving the network, more generalized to being what goal you’re trying to achieve.

## Videos

[Demonstration 1](https://www.youtube.com/watch?v=R__f9THwd-A&feature=youtu.be)

