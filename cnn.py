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
        model.add(MaxPooling3D(pool_size=tuple([nb_pool[0]]*3)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, init='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.labels), init='normal'))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='RMSprop')
        return model


    def train(self, data):
        X_tr = []
        Y_tr = []

        for video in data.get_training():
            Y_tr.append(video.info.type)
            X_tr.append(self.preprocess_video(video))


        self.labels = sorted(set(Y_tr))

        y_train = np.array([self.labels.index(x) for x in Y_tr])
        X_train = np.array(X_tr)

        num_samples = len(X_tr) 

        train_set = np.zeros((num_samples, 1, self.resize[0],self.resize[1],self.resize[2]))

        for h in xrange(num_samples):
            train_set[h][0][:][:][:]=X_train[h,:,:,:]
          

        # CNN Training parameters

        batch_size = 2
        nb_classes = 6
        nb_epoch = 3

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        train_set = train_set.astype('float32')
        train_set -= np.mean(train_set)
        train_set /= np.max(train_set)



        self.model = self.get_model()
        self.model.fit(train_set, Y_train, 
                  batch_size=batch_size, nb_epoch=nb_epoch,
                  show_accuracy=True,shuffle=True)


    def test(self, data):
        X_te = []
        Y_te = []

        for video in data.get_test():
            Y_te.append(video.info.type)
            X_te.append(self.preprocess_video(video))


        y_test = np.array([self.labels.index(x) for x in Y_te])
        X_test = np.array(X_te)

        num_samples = len(X_te) 

        test_set = np.zeros((num_samples, 1, self.resize[0],self.resize[1],self.resize[2]))

        for h in xrange(num_samples):
            test_set[h][0][:][:][:] = X_test[h,:,:,:]
          

        # convert class vectors to binary class matrices
        test_set = test_set.astype('float32')
        test_set -= np.mean(test_set)
        test_set /= np.max(test_set)

        pred = self.model.predict(test_set, batch_size=2)
        pred = [np.argmax(x) for x in pred]
        act = y_test

        print confusion_matrix(act, pred)

data = DataSet("dataset")
v = VideoClassifier(resize=(32,32,15))
v.train(data)
v.test(data)

