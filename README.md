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

The convolutional neural network used for this project is a network published by NVIDIA. This network is designed for the 
End-to-End deep learning for self driving cars. More details about this network can be found [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).

This network consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers, as shown
in the figure below:

![NVIDIA_CNN](examples/cnn-architecture-624x890.png)
Source: https://devblogs.nvidia.com/deep-learning-self-driving-cars/

In the model python file [bcn_model.py](bcn_model.py), line 103 is where the normalization takes place.

```
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
```

The input images are normalized between -0.5 and +0.5. Additionally, 2D cropping is also done.

```
model.add(Cropping2D(cropping=((70,25),(0,0))))
```

From the top of the image, 70 pixels and from the bottom, 25 pixels are removed horizontally. This is done so that the network
does not get distracted by unnecessary objects in the scene and the focus of learning remains on the track.

The model in the code as shown in the above figure can be found between lines 106 and 116.

```
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

To introduce non linearity to the model, ReLU activation is used.