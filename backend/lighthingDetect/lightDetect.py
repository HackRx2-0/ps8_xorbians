import cv2
import numpy as np

def low_light_detection(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    x = np.mean(img[:, :, 2])
    z = x*(-0.14439479)+9.92969547
    y_pred = 1/(1+np.exp(-z))
    if y_pred < 0.5:
        return (False, 1-y_pred)
    else:
        return (True, y_pred) 

def high_light_detection(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    x = np.mean(img[:, :, 2])
    z = x*(0.075)-16.875
    y_pred = 1/(1+np.exp(-z))
    if y_pred < 0.5:
        return (False, 1-y_pred)
    else:
        return (True, y_pred) 