import os
import sys

from data import DataSet
from sift_classifier import train_test as train_test_sift
from cnn import train_test as train_test_cnn


def main():
    d = DataSet("dataset")
    
    if sys.argv[1:] and sys.argv[1] == 'sift':
        print train_test_sift(d)
    else:
        print train_test_cnn(d)

if __name__ == '__main__':
    main()

