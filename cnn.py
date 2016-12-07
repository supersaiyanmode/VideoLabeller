from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D

from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils

import theano
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from data import DataSet


class VideoClassifier(object):
    def __init__(self, resize=(16,16,15)):
        self.resize = list(resize)
    
    def preprocess_video(self, frames):
        r = lambda x: cv2.resize(x, tuple(self.resize[:2]), 
                                interpolation=cv2.INTER_AREA)
        frames = [r(x) for x in frames[:self.resize[2]]]
        inp = np.array(frames)
        return np.rollaxis(np.rollaxis(inp, 2, 0), 2, 0)


    def get_model(self):
        nb_filters = [32, 32]
        nb_pool = [3, 3]
        nb_conv = [5,5]

        # Define model
        model = Sequential()
        model.add(Convolution3D(
            nb_filters[0],
            nb_depth=nb_conv[0],
            nb_row=nb_conv[0],
            nb_col=nb_conv[0],
            input_shape=tuple([1] + self.resize),
            activation='relu'))
        model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, init='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.labels), init='normal'))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='RMSprop')
        return model


    def train(self, data):
        X_tr=[]           # variable to store entire dataset
        Y_tr = []
        labels = set()

        for video in data.get_training():
            labels.add(video.info.type)
            Y_tr.append(video.info.type)

            frames =[cv2.resize(x,(self.resize[0],self.resize[1]),interpolation=cv2.INTER_AREA) for x in video.frames[:15]]
            inp = np.array(frames)
            inp = np.rollaxis(np.rollaxis(inp, 2, 0), 2, 0)
            X_tr.append(inp)

        self.labels = sorted(labels)

        X_tr_array = np.array(X_tr)   # convert the frames read into array
        label = np.array([self.labels.index(x) for x in Y_tr])

        num_samples = len(X_tr_array) 

        train_data = [X_tr_array,label]

        (X_train, y_train) = (train_data[0],train_data[1])
        print('X_Train shape:', X_train.shape)

        train_set = np.zeros((num_samples, 1, self.resize[0],self.resize[1],self.resize[2]))

        for h in xrange(num_samples):
            train_set[h][0][:][:][:]=X_train[h,:,:,:]
          

        print(train_set.shape, 'train samples')

        # CNN Training parameters

        batch_size = 2
        nb_classes = 6
        nb_epoch = 3

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        train_set = train_set.astype('float32')
        train_set -= np.mean(train_set)
        train_set /=np.max(train_set)



        model = self.get_model()
        # Split the data

        #X_train_new, X_val_new, y_train_new,y_val_new =  train_test_split(train_set, Y_train, test_size=0.2, random_state=4)


        # Train the model
        #print X_train_new.shape, y_train_new.shape

        import pdb; pdb.set_trace()
        hist = model.fit(train_set, Y_train, 
                  batch_size=batch_size,nb_epoch = nb_epoch,show_accuracy=True,shuffle=True)

        #hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new,y_val_new),
        #          batch_size=batch_size,nb_epoch = nb_epoch,show_accuracy=True,shuffle=True)


        #hist = model.fit(train_set, Y_train, batch_size=batch_size,
        #         nb_epoch=nb_epoch,validation_split=0.2, show_accuracy=True,
        #           shuffle=True)


         # Evaluate the model
        print X_val_new.shape, y_val_new.shape
        #y_val_pred = model.predict(X_val_new, batch_size=batch_size, show_accuracy=True)
        y_val_pred = model.predict(X_val_new, batch_size=batch_size)
        pred = list(np.argmax(x) for x in y_val_pred)
        act = list(np.argmax(x) for x in y_val_new)
        print confusion_matrix(act, pred)

data = DataSet("dataset")
v = VideoClassifier()
v.train(data)
