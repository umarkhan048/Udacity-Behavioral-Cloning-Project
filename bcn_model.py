import csv
import cv2
import numpy as np

lines = []

with open('../../../opt/TrainingData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
firstline = True
for line in lines:
    if firstline:
        firstline = False
        continue
    source_path = line[0]
    source_path_left = line[1]
    source_path_right = line[2]
    
    filename_center = source_path.split('/')[-1]
    filename_left = source_path_left.split('/')[-1]
    filename_right = source_path_right.split('/')[-1]
    center_path = '../../../opt/TrainingData/IMG/' + filename_center
    left_path = '../../../opt/TrainingData/IMG/' + filename_left
    right_path = '../../../opt/TrainingData/IMG/' + filename_right
    image_center = cv2.imread(center_path)
    image_left = cv2.imread(left_path)
    image_right = cv2.imread(right_path)
    image_center_flipped = np.fliplr(image_center)
    images.append(image_center)
    images.append(image_center_flipped)
    images.append(image_left)
    images.append(image_right)
    measurement_center = float(line[3])
    correction_factor = 0.2
    measurement_left = measurement_center + correction_factor
    measurement_right = measurement_center - correction_factor
        
    measurement_center_flipped = -measurement_center
    measurements.append(measurement_center)
    measurements.append(measurement_center_flipped)
    measurements.append(measurement_left)
    measurements.append(measurement_right)
    
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
#model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,5,5, activation="relu"))
model.add(Convolution2D(64,5,5, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
model.save('model.h5')