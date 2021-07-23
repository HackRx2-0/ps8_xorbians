import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model = load_model("./checkReal/realFake.model")

def check_real(image):
    # image = cv2.imread(img)
    image = cv2.resize(image, (32, 32))
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    data = []
    data.append(image)
    data = np.array(data, dtype="float") / 255.0

    prediction = model.predict(x=data)
    if prediction[0][1] > prediction[0][0]:
        print(prediction[0][1])
        return (False,prediction[0][1])
    print(prediction[0][0])
    return (True,prediction[0][1])