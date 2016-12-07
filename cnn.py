"""
    Majority of model construction portions come from:
        - https://github.com/MinhazPalasara/keras/blob/master/examples/shapes_3d_cnn.py
        - http://learnandshare645.blogspot.com/2016/06/3d-cnn-in-keras-action-recognition.html
"""


import pickle
import json

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
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
import config


class VideoClassifier(object):
    def __init__(self, resize=(16,16,15)):
        self.resize = list(resize)
    

    def preprocess_video(self, frames):
        r = lambda x: cv2.resize(x, tuple(self.resize[:2]))
        frames = [r(x) for x in frames[:self.resize[2]]]
        inp = np.array(frames)
        return np.rollaxis(np.rollaxis(inp, 2, 0), 2, 0)


    def get_model1(self):
        nb_filters = [64]
        nb_pool = [3]
        nb_conv = [7]

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
        model.add(Flatten())
        model.add(Dropout(0.5))

        model.add(Dense(128, init='normal', activation='relu'))
        model.add(Dense(32, init='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.labels), init='normal'))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='RMSprop')
        return model


    def get_model(self):
        """This is almost a copy-paste from the link mentioned in the
        head of the page."""

	# number of convolutional filters to use at each layer
	nb_filters = [16, 32, 48]

	# level of pooling to perform at each layer (POOL x POOL)
	nb_pool = [2, 2, 2]

	# level of convolution to perform at each layer (CONV x CONV)
	nb_conv = [5, 5, 4]

	model = Sequential()
	model.add(Convolution3D(nb_filters[0],nb_depth=nb_conv[0], nb_row=nb_conv[0], nb_col=nb_conv[0], border_mode='full',
            input_shape=tuple([1] + self.resize)))
	model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
	model.add(Dropout(0.5))

	model.add(Convolution3D(nb_filters[1],nb_depth=nb_conv[1], nb_row=nb_conv[1], nb_col=nb_conv[1], border_mode='full',
				activation='relu'))
	model.add(MaxPooling3D(pool_size=(nb_pool[1], nb_pool[1], nb_pool[1])))
	model.add(Dropout(0.5))

	model.add(Convolution3D(nb_filters[2],nb_depth=nb_conv[2], nb_row=nb_conv[2], nb_col=nb_conv[2], border_mode='full',
				activation='relu'))
	model.add(MaxPooling3D(pool_size=(nb_pool[2], nb_pool[2], nb_pool[2])))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(512, init='normal', activation='relu'))
	model.add(Dense(6, init='normal'))
	model.add(Activation('softmax'))

	sgd = RMSprop(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer='RMSprop')
	return model


    def train(self, data, epochs=3):
        X = []
        Y = []

        for video in data.get_training():
            Y.append(video.info.type)
            X.append(self.preprocess_video(video))

        self.labels = sorted(set(Y))

        Y = np.array([self.labels.index(x) for x in Y])
        X = np.array(X)

        x = np.zeros((len(X), 1, self.resize[0],self.resize[1],self.resize[2]))

        for h in xrange(len(X)):
            x[h][0][:][:][:] = X[h,:,:,:]

        X = x.astype('float32')
        Y = np_utils.to_categorical(Y, len(self.labels))
        X -= np.mean(X)
        X /= np.max(X)

        self.model = self.get_model1()
        self.model.fit(X, Y, 
                  batch_size=2, nb_epoch=epochs,
                  show_accuracy=True,shuffle=True)


    def test(self, data):
        X = []
        Y = []

        for video in data.get_test():
            Y.append(video.info.type)
            X.append(self.preprocess_video(video))

        Y = np.array([self.labels.index(x) for x in Y])
        X = np.array(X)

        x = np.zeros((len(X), 1, self.resize[0],self.resize[1],self.resize[2]))

        for h in xrange(len(X)):
            x[h][0][:][:][:] = X[h,:,:,:]
          
        X = x.astype('float32')
        X -= np.mean(X)
        X /= np.max(X)

        pred = self.model.predict(X, batch_size=2)
        pred = [np.argmax(x) for x in pred]
        act = Y

        return act, pred


    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

        self.model.save_weights(path + "/classifier-weights")

        with open(path + "/classifier-struct", "w") as f:
            json.dump(self.model.to_json(), f)

        with open(path + "/data", "w") as f:
            json.dump({"resize": self.resize, "labels": self.labels}, f)


    @staticmethod
    def load(path):
        with open(path + "/classifier-struct") as f:
            obj = json.load(f)
            model = model_from_json(obj)
            model.load_weights(path + "/classifier-weights")

        with open(path + "/data") as f:
            obj = json.load(f)

        v = VideoClassifier(obj["resize"])
        v.model = model
        v.labels = obj["labels"]

        return v

def train_test(data):
    if config.save_model and os.path.exists(config.save_model):
        v = VideoClassifier.load(config.save_model)
        print "****LOAD MODEL****"
    else:
        v = VideoClassifier(resize=(32,32,24))
        v.train(data)

        if config.save_model:
            print "***SAVE MODEL****"
            v.save(config.save_model)
    return confusion_matrix(*v.test(data))

