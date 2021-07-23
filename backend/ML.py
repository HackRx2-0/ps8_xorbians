from opencv_dnn.detect import Detect
from statical_methods.blur import check_blur
from statical_methods.size import check_dimensions
from checkReal.realFakeDetect import check_real
from fastapi.encoders import jsonable_encoder
import cv2
import numpy as np
import time
import os

def ML(path):
	path_OG = np.array(path)
	path = cv2.cvtColor(path_OG, cv2.COLOR_RGB2BGR)

	# print(path)
	# print(path.shape)
	start = time.time()
	save = True
	issues = []
	score = []
	d = Detect()
	original, detected, bbox, confidences = d.detect_in_image(path)
	try:
		confidenceDetect = sum(confidences)/len(confidences)
	except:
		confidenceDetect = float(0)
	score.append(float(confidenceDetect))
	if save:
		cv2.imwrite('./output/detected.jpg', detected)
	if len(bbox)>1:
		issues.append("More than one face detected - Total number = "+str(len(bbox)))
		bbox = bbox[0]
	elif len(bbox)==1:
		bbox = bbox[0]
	else:
		issues.append("No faces detected")
		out = jsonable_encoder({"output":[{'issues':issues},{'score':[0]}]})
		return out
	
	x,y,x1,y1 = bbox
	img = original[y:y1, x:x1, :]
	
	sobel, blur, blurScore = check_blur(img)
	if save:
		cv2.imwrite('./output/sobel_edge.jpg', sobel)
	if blur:
		issues.append("Face is Blurred")
		score.append(float(1-float(blurScore)))
	else:
		score.append(float(blurScore))
	resized, small = check_dimensions(original, img)
	if save:
		cv2.imwrite('./output/resized.jpg', resized)
	if small:
		issues.append("Too small")

	fake,score_realFake = check_real(resized[:,:,:])
	if fake:
		issues.append("Fake image")
	score.append(float(score_realFake))
	end = time.time()
	# print("time : ",end-start)
	out = jsonable_encoder({"output":[{'issues':issues},{'score':score}]})
	return out

if __name__ == '__main__':
	path = './images/'
	for img_path in [path+i for i in sorted(os.listdir(path))]:
		ML(img_path)