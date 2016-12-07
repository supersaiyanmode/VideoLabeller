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


class VideoClassifier(object):
    def __init__(self, resize=(16, 16, 15)):
        self.resize = list(resize)

    def preprocess(self, frames):
        frames = frames[:self.resize[2]]
        frames = [cv2.resize(x, tuple(self.resize[:2])) for x in frames]
        inp = np.array(frames)
        return np.rollaxis(np.rollaxis(inp, 2, 0), 2, 0)
        

    def train(self, data):
        labels = set(x.info.type for x in data.get_training())
        min_frames = min(x.total_frames for x in data.get_training())
        min_frames = min(min_frames, (x.total_frames for x in data.get_test()))

        self.resize[2] = min(min_frames, self.resize[2])
        self.labels = sorted(labels)

        features, target = [], []
        for video in data.get_training():
            inp = self.preprocess(video.frames)
            features.append(inp)
            target.append(self.labels.index(video.info.type))
        
        train_set = np.zeros((len(features), 1, self.resize[0], self.resize[1],
                             self.resize[2]))

        X = np.array(features)
        for h in xrange(len(features)):
            train_set[h][0][:][:][:]=X[h,:,:,:]
        train_set = train_set.astype('float32')
        train_set -= np.mean(train_set)
        train_set /= np.max(train_set)

        Y = np_utils.to_categorical(np.array(target), len(labels))

        model = self.get_model()
        import pdb; pdb.set_trace()
        return model.fit(train_set, Y, 
                batch_size=2, nb_epoch=50,
                show_accuracy=True, shuffle=True)
    
    def test(self, dataset):
        actual, pred = [], []

        for video in dataset.get_test():
           pass 

    def get_model(self):
        filters = [32, 32]
        pool = [3, 3]
        conv = [5, 5]

        model = Sequential()
        model.add(Convolution3D(
            filters[0],
            nb_depth=conv[0], nb_row=conv[0], nb_col=conv[0],
            input_shape=(1, self.resize[0], self.resize[1], self.resize[2]),
            activation='relu'))

        model.add(MaxPooling3D(pool_size=(pool[0], pool[0], pool[0])))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, init='normal', activation='relu'))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='RMSprop')

        return model
        
def train_test(dataset):
    classifier = VideoClassifier()
    classifier.train(dataset)
    return classifier.test(dataset)


