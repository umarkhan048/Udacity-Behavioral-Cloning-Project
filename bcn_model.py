import csv
import cv2
import numpy as np
import math

samples =[]
firstline = True
# Reading in the csv file
with open('TrainingData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader: 
        if firstline:
            firstline = False
            continue
        samples.append(line)

from sklearn.model_selection import train_test_split
import sklearn
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            
            for batch_sample in batch_samples:
                
                # from the input csv training data file, source paths for center, left
                # and right side cameras are read in
                source_path = batch_sample[0]
                source_path_left = batch_sample[1]
                source_path_right = batch_sample[2]
                
                # Based on the above paths, filenames are extracted. The files are
                # present in the IMG folder of the training data
                filename_center = source_path.split('/')[-1]
                filename_left = source_path_left.split('/')[-1]
                filename_right = source_path_right.split('/')[-1]
                
                # Creating paths of images
                center_path = 'TrainingData/IMG/' + filename_center
                left_path = 'TrainingData/IMG/' + filename_left
                right_path = 'TrainingData/IMG/' + filename_right
    
                # Reading in images using openCV
                image_center = cv2.imread(center_path)
                image_left = cv2.imread(left_path)
                image_right = cv2.imread(right_path)
                
                # Augmenting center camera data by flipping the images and appending
                # to the images array
                image_center_flipped = np.fliplr(image_center)
                images.append(image_center)
                images.append(image_center_flipped)
                images.append(image_left)
                images.append(image_right)

                # Steering angle from the perspective of the center camera is read directly
                measurement_center = float(line[3])
    
                # Steering angle from the perspecitve of the left and right cameras must be
                # corrected with a factor. The factor is tunable
                correction_factor = 0.2
                measurement_left = measurement_center + correction_factor
                measurement_right = measurement_center - correction_factor
   
                # Like the image, the steering angle corresponding to the flipped center
                # image must also be corrected
                measurement_center_flipped = -measurement_center
    
                # All the measurements are appened to the measurements array
                measurements.append(measurement_center)
                measurements.append(measurement_center_flipped)
                measurements.append(measurement_left)
                measurements.append(measurement_right)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

# Creating a model from Keras
model = Sequential()
# Normalizaiton of the input data
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
# Nvidia architecture is implemented with ReLu activation layers
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

# Mean Squared Error metric is used to judge performance and adam
# optimizer for backprop
adm = Adam(lr=0.0005)
model.compile(loss='mse', optimizer=adm)
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
                    validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), 
                    epochs=7, verbose=1)
model.save('model.h5')