import cv2

fourcc = cv2.VideoWriter_fourcc(*'XVID')

def convert(path, target):
    capture = cv2.VideoCapture(path)
    img = None
    writer = cv2.VideoWriter(target, fourcc, 24, ..)
