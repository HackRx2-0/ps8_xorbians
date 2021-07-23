import cv2
import numpy as np
import os

class Detect:
    def __init__(self):
        self.check_setup()
        self.modelFile = "./opencv_dnn/models/res10_300x300_ssd_iter_140000.caffemodel"
        self.configFile = "./opencv_dnn/models/deploy.prototxt.txt"
        self.net = cv2.dnn.readNetFromCaffe(self.configFile, self.modelFile)

    def check_setup(self):
        if not os.path.exists("./opencv_dnn/models"):
            os.mkdir("./opencv_dnn/models")
        if not os.path.exists("./opencv_dnn/models/deploy.prototxt.txt"):
            os.system("curl https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt > ./opencv_dnn/models/deploy.prototxt.txt")
            os.system("curl https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel > ./opencv_dnn/models/res10_300x300_ssd_iter_140000.caffemodel")

    def detect_in_image(self, img):
        #img = cv2.imread(path)
        orig = img.copy()
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
        (300, 300), (104.0, 117.0, 123.0))
        self.net.setInput(blob)
        faces = self.net.forward()
        boxes = []
        confidences = []
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
                boxes.append([x, y, x1, y1])
                confidences.append(confidence)
        return orig, img, boxes, confidences

if __name__ == '__main__':
    d = Detect()
    print(d.detect_in_image())