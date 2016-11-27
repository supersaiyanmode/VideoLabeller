import random
from collections import Counter
from collections import defaultdict

from sklearn import svm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

from descriptor import get_descriptor
from data import DataSet
from descriptor_cache import Cache
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

        cache[str(video)] = descriptors
        yield video.info, descriptors
    
        if count % 100 == 0:
            print "***WRITING CACHE***"
            cache.write()
    print "***WRITING CACHE***"
    cache.write()

class VideoClassifier(object):
    def __init__(self, classifier):
        self.cache = Cache()
        self.classifier = classifier
        self.multilabel = MultiLabelBinarizer()

    def train(self, dataset):
        features, target = [], []

        samples = get_samples(dataset.get_training(), self.cache)
        for info, descriptors in samples:
            for desc in descriptors:
                features.append(desc)
                target.append(info.type)
        
        X = features
        Y = self.multilabel.fit_transform([[x] for x in target])
        
        self.kmeans = KMeans(n_clusters=8, verbose=True)
        newX = self.kmeans.fit_transform(X)


        self.classifier.fit(newX, Y)

    def test(self, dataset):
        actual, pred = [], []

        samples = get_samples(dataset.get_test(), self.cache)
        for info, descriptors in samples:
            X = self.kmeans.transform(descriptors)
            Y = self.classifier.predict(X)
            Y = [x[0] for x in self.multilabel.inverse_transform(Y)]
            c = Counter(Y)
            actual.append(info.type)
            pred.append(c.most_common()[0][0])
        return actual, pred

def main():
    d = DataSet("dataset", train_file="train-small-4.txt",
                test_file="test-small-4.txt")
    #d = DataSet("dataset")
    params = {
        "decision_function_shape": "ovr",
        "verbose": True,
    }
    c = OneVsRestClassifier(svm.SVC(**params))
    classifier = VideoClassifier(c)
    classifier.train(d)
    actual, pred = classifier.test(d)

    print confusion_matrix(actual, pred)

if __name__ == '__main__':
    main()

