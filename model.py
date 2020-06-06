#import libraries  
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %tensorflow_version 1.x
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, Lambda
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random
import utils


#set up data dir
datadir = '/home/workspace/CarND-Behavioral-Cloning-P3/data'
#read csv file
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'))

pd.set_option('display.max_colwidth', -1)
data.head()

#split head and tail for first three data column
def path_leaf(path):
  head, tail = ntpath.split(path)
  return tail
data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
data.head()
num_bins = 25
samples_per_bin = 700
hist, bins = np.histogram(data['steering'], num_bins)


#limit the amount of 0 steering angle data in the dataset
remove_list = []
for j in range(num_bins):
  list_ = []
  for i in range(len(data['steering'])):
    if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
      list_.append(i)
  list_ = shuffle(list_)
  list_ = list_[samples_per_bin:]
  remove_list.extend(list_)
 
data.drop(data.index[remove_list], inplace=True)

#split data set
def load_img_steering(datadir, df):
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    image_path.append(os.path.join(datadir, center.strip()))
    steering.append(float(indexed_data[3]))
    # left image append
    image_path.append(os.path.join(datadir,left.strip()))
    #offset steering angle
    steering.append(float(indexed_data[3])+0.15)
    # right image append
    image_path.append(os.path.join(datadir,right.strip()))
    #offset steering angle
    steering.append(float(indexed_data[3])-0.15)
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings

image_paths, steerings = load_img_steering(datadir + '/IMG', data)

#split traning & validation data set
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)

#randomly augment the training data 
def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
      image = utils.pan(image)
    if np.random.rand() < 0.5:
      image = utils.zoom(image)
    if np.random.rand() < 0.5:
      image = utils.img_random_brightness(image)
    if np.random.rand() < 0.5:
      image, steering_angle = utils.img_random_flip(image, steering_angle)
    
    return image, steering_angle

#preprocess images
def img_preprocess(img):
    #cropping up the interested area in images 
    img = img[60:135,:,:]
    #apply GaussianBlur filter
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    #resize image
    img = cv2.resize(img, (200, 66))
    #normalizing each image data 
    img = img/255
    return img

#image generator
def batch_generator(image_paths, steering_ang, batch_size, istraining):
  
  while True:
    batch_img = []
    batch_steering = []
    
    for i in range(batch_size):
      random_index = random.randint(0, len(image_paths) - 1)
      
      if istraining:
        im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
     
      else:
        im = mpimg.imread(image_paths[random_index])
        steering = steering_ang[random_index]
      
      im = img_preprocess(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield (np.asarray(batch_img), np.asarray(batch_steering))
#build Nvidia deepling learning model
def nvidia_model():
  model = Sequential()
  #model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(66, 200, 3)))
  #Convolution 5x5
  model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2),input_shape=(66, 200, 3)))
  #ELU activation
  model.add(Activation('elu'))
  #Max pooling
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

  #Convolution 5x5
  model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
  model.add(Activation('elu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
  #Convolution 5x5
  model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
  model.add(Activation('elu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

  #Convolution 3x3
  model.add(Convolution2D(64, 3, 3, activation='elu'))
  #Convolution 3x3
  model.add(Convolution2D(64, 3, 3, activation='elu'))
  #model.add(Dropout(0.5))
  
  #Flatten layer
  model.add(Flatten())
  #Fully Connected layer
  model.add(Dense(1000, activation = 'elu'))
  #dropout layer
  model.add(Dropout(0.5))
  #Fully Connected layer
  model.add(Dense(500, activation = 'elu'))
  #Fully Connected layer
  model.add(Dense(100, activation = 'elu'))
  #model.add(Dropout(0.5))
  #Fully Connected layer
  model.add(Dense(50, activation = 'elu'))
  #model.add(Dropout(0.5))
  #Fully Connected layer
  model.add(Dense(10, activation = 'elu'))
  #model.add(Dropout(0.5))
  #Fully Connected layer
  model.add(Dense(1))
  
  optimizer = Adam(lr=0.0009)
  model.compile(loss='mse', optimizer=optimizer)
  return model

model = nvidia_model()
#summary model archtecture
model.summary()
#fit generator
history = model.fit_generator(batch_generator(X_train, y_train, 100, 1),
                                  steps_per_epoch=300, 
                                  epochs=15,
                                  validation_data=batch_generator(X_valid, y_valid, 100, 0),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle = 1)


#save deep learning model in workspace
model.save('model.h5', overwrite=True)

