# Behavioral Cloning Project



Overview
---

The project "Behavioral Cloning" is based on the idea of end-to-end learning of an autonomous vehicle. It means that input
to the vehicle is the data from sensors (which in this case are three front cameras placed in the center, left and right side
of the vehicle). The training data is gathered from a simulation environment. The output of the simulation are the images from 
the center, left and right cameras as well as steering corresponding steering angle and speed. For the purpose of this project,
only the steering angle is of interest. Based on the training data i.e. scenarios and the steering angle in that particular
scenario, the network learns to predict a correct steering angle for each situation. The goal is to keep the car on the drivable
track in the autonomous driving mode. The main idea behind it is, in contrast to other approaches, that not only perception, but
also interpretation, motion planning and control part is also learned by the vehicle. In essence, between the part between sensor
input data and the steering commands is a black-box.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

Model Architecture and Training Strategy
---

