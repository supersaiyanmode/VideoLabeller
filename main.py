import random
from collections import Counter
from collections import defaultdict

from sklearn import svm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix

from descriptor import get_descriptor
from data import DataSet
import config


def generate_coord(shape):
    seen = set()
    
    while True:
        coord = tuple([random.randrange(0, d) for d in shape])
        if coord in seen:
            continue

        seen.add(coord)
        yield coord

def get_samples(videos):
    for video in videos:
        descriptors = []
        print "Training", video, "Shape:", video.shape()
        for coord in generate_coord(video.shape()):
            coord = list(coord)
            #coord[2] += video.start
            print "  Descriptor at:", coord,
            desc = get_descriptor(video, coord)
            if not desc:
                print "[BAD]"
                continue
            print "[GOOD]"
            descriptors.append(desc)

            if len(descriptors) >= config.training_points:
                break

        yield video.info, descriptors

class VideoClassifier(object):
    def __init__(self, classifier):
        self.classifier = OneVsRestClassifier(classifier)
        self.multilabel = MultiLabelBinarizer()

    def train(self, dataset):
        features, target = [], []

        for info, descriptors in get_samples(dataset.get_training()):
            for desc in descriptors:
                features.append(desc)
                target.append(info.type)
        
        X = features
        Y = self.multilabel.fit_transform([[x] for x in target])
        self.classifier.fit(X, Y)

    def test(self, dataset):
        actual, pred = [], []
        for info, descriptors in get_samples(dataset.get_test()):
            X = descriptors
            Y = self.classifier.predict(X)
            Y = [x[0] for x in self.multilabel.inverse_transform(Y)]
            c = Counter(Y)
            actual.append(info.type)
            pred.append(c.most_common()[0][0])
        return actual, pred

def main():
    d = DataSet("dataset")
    classifier = VideoClassifier(svm.SVC(verbose=True))
    classifier.train(d)
    actual, pred = classifier.test(d)

    print confusion_matrix(actual, pred)

if __name__ == '__main__':
    main()

