import os

from data import DataSet
import config
from sift_classifier import train_test as train_test_sift
from cnn_classifier import train_test as train_test_cnn


def main():
    d = DataSet("dataset", train_file="train-small-1.txt",
                test_file="test-small-1.txt")
    d = DataSet("dataset")
    
    if True:
        train_test_cnn(d)
    else:
        train_test_sift(d)

    print confusion_matrix(actual, pred)

if __name__ == '__main__':
    main()

