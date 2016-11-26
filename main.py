import random
from collections import Counter
from collections import defaultdict

from sklearn import svm
from sklearn.preprocessing import MultiLabelBinarizer

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
            coord[2] += video.start
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
        self.classifier = classifier
        self.multilabel = MultiLabelBinarizer()

    def train(self, dataset):
        X, Y = [], []

        for info, descriptors in get_samples(dataset.get_training()):
            for desc in descriptors:
                X.append(desc)
                Y.append(info.type)
        
        Y = self.multilabel.fit_trasform([x] for x in Y)

        self.classifier.fit(X, Y)

    def test(self, dataset):
        res = []
        for info, descriptors in get_samples(dataset.get_test()):
            X = descriptors
            Y = [suppress(x) for x in self.classifier.predict(X)]
            Y = [x[0] for x in self.multilabel.inverse_transform(Y)]
            c = Counter(Y)
            res.append((info.type, c.most_common()))
        return res

def main():
    d = DataSet("dataset")
    classifier = VideoClassifier(svm.SVC(verbose=True))
    classifier.train(d)

    confusion_matrix = defaultdict(lambda: defaultdict(int))
    for actual, predicted in classifier.test(d):
        confusion_matrix[actual][predicted] += 1
    print confusion_matrix

if __name__ == '__main__':
    main()

