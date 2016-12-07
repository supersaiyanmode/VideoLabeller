import random
from collections import Counter
from collections import defaultdict
import pickle

from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from descriptor import get_descriptor
from descriptor_cache import Cache
from utils import normalize
import config


def generate_coord(shape):
    seen = set()
    
    while True:
        coord = tuple([random.randrange(0, d) for d in shape])
        if coord in seen:
            continue

        seen.add(coord)
        yield coord


def get_samples(videos, cache):
    count = 0
    for video in videos:
        descriptors = []
        print "Processing: ", video, "Shape:", video.shape()

        if str(video) in cache:
            print "  (Cached)"
            yield video.info, cache[str(video)]
            continue

        count += 1

        for coord in generate_coord(video.shape()):
            coord = list(coord)
            print "  Descriptor at:", coord,
            desc = get_descriptor(video, coord)
            if not desc:
                print "[BAD]"
                continue
            print "[GOOD]"
            descriptors.append(desc)

            if len(descriptors) >= config.training_points:
                break

        print "Extracted", count, "descriptors"

        cache[str(video)] = descriptors
        yield video.info, descriptors
    
        if count % 100 == 0:
            print "***WRITING CACHE***"
            cache.write()
    if count:
        print "***WRITING CACHE***"
        cache.write()

class VideoClassifier(object):
    def __init__(self, classifier, kmeans=None):
        self.cache = Cache()
        self.classifier = classifier
        self.kmeans = kmeans or KMeans(n_clusters=8, verbose=True)

    def save(self, path):
        
        if not os.path.exists(path):
            os.mkdir(path)

        with open(path + "/classifier", "w") as f:
            pickle.dump(self.classifier, f)
        with open(path + "/kmeans", "w") as f:
            pickle.dump(self.kmeans, f)

    @staticmethod
    def load(path):
        with open(path + "/classifier") as f:
            classifier = pickle.load(f)
        with open(path + "/kmeans") as f:
            kmeans = pickle.load(f)
        return VideoClassifier(classifier, kmeans)

    def post_process(self, vec):
        return normalize(vec)

    def train(self, dataset):
        features, target = [], []

        samples = get_samples(dataset.get_training(), self.cache)
        for info, descriptors in samples:
            for desc in descriptors:
                features.append(self.post_process(desc))
                target.append(info.type)
        
        X = features
        Y = target
        
        newX = self.kmeans.fit_transform(X)
        
        self.classifier.fit(newX, Y)

    def test(self, dataset):
        actual, pred = [], []

        samples = get_samples(dataset.get_test(), self.cache)
        for info, descriptors in samples:
            X = self.kmeans.transform(map(self.post_process, descriptors))
            Y = self.classifier.predict(X)
            c = Counter(Y)
            actual.append(info.type)
            pred.append(c.most_common()[0][0])
            print confusion_matrix(actual, pred)
        return actual, pred

def train_test(datset):
    params = {
        "decision_function_shape": "ovr",
        "verbose": True,
    }

    if config.save_model and os.path.exists(config.save_model):
        classifier = VideoClassifier.load(config.save_model)
        print "****LOAD MODEL****"
    else:
        c = OneVsRestClassifier(svm.SVC(**params))
        classifier = VideoClassifier(c)
        classifier.train(d)

        if config.save_model:
            print "***SAVE MODEL****"
            classifier.save(config.save_model)
    
    actual, pred = classifier.test(d)
    return actual, pred

