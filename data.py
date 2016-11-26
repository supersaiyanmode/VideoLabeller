import os.path
import random
import cv2

class SampleInfo(object):
    def __init__(self, basedir, line):
        parts = [x.strip() for x in line.strip().split("\t")]
        self.type = self.get_type(parts[0])
        self.path = self.get_path(basedir, parts[0])
        self.frames = self.parse_frames(parts[2])
        if not os.path.isfile(self.path):
            raise Exception("Data file not found: " + self.path)

    def get_type(self, name):
        return name.split("_")[1]

    def get_path(self, basedir, name):
        return basedir + "/" + self.get_type(name) + "/" + name + "_uncomp.avi"
    
    def parse_frames(self, line):
        return [tuple(map(int, x.strip().split("-"))) for x in line.split(", ")]

    def __str__(self):
        return "<SampleInfo %s (%s) %s>"%(self.type, self.path, self.frames)

class VideoSample(object):
    def __init__(self, capture, sample, start, end):
        self.frames = []
        self.info = sample
        self.start = start
        self.end = end
        self.total_frames = self.end - self.start + 1

        capture.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
        for _ in xrange(start, end+1):
            self.frames.append(self.preprocess(capture.read()[1]))
        self.height, self.width = self.frames[0].shape
    
    def shape(self):
        return self.height, self.width, self.total_frames

    def preprocess(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def play(self, window_name, wait=3):
        for frame in self.frames:
            cv2.imshow(window_name, frame)
            cv2.waitKey(wait)

    def __getitem__(self, key):
        return self.frames[key]

    def __str__(self):
        return "<VideoSample (%s, %s), frames: %d -> %d"%(self.info.type,
                self.info.path, self.start, self.end)

class DataSet(object):
    def __init__(self, datadir, train_file="train.txt", test_file="test.txt"):
        self.datadir = datadir
        self.train = [SampleInfo(datadir, x) for x in open(datadir + "/" + train_file)]
        self.test = [SampleInfo(datadir, x) for x in open(datadir + "/" + test_file)]
    
    def get_training(self):
        for sample in self.train:
            for video_sample in self.get_frames(sample):
                yield video_sample
    
    def get_test(self):
        for sample in self.test:
            for video_sample in self.get_frames(sample):
                yield video_sample

    def get_frames(self, sample):
        capture = cv2.VideoCapture(sample.path)
        for frame_seq in sample.frames:
            yield VideoSample(capture, sample, frame_seq[0], frame_seq[1])


if __name__ == '__main__':
    d = DataSet("dataset")
    cv2.namedWindow("x")
    cv2.resizeWindow("x", 500, 500)

    for video_sample in d.get_training():
        print "Showing: ", video_sample
        video_sample.play("x")
    cv2.destroyWindow("x")
        
