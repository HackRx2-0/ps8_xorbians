from opencv_dnn.detect import Detect
from statical_methods.blur import check_blur
from statical_methods.size import check_dimensions
from checkReal.realFakeDetect import check_real
from checkObstruction.maskDetect import check_obstruct
from lighthingDetect.lightDetect import low_light_detection
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
	name = []
	d = Detect()
	original, detected, bbox, confidences = d.detect_in_image(path)
	try:
		confidenceDetect = sum(confidences)/len(confidences)
	except:
		confidenceDetect = float(0)
	score.append(float(confidenceDetect))
	name.append("confidence")
	if save:
		cv2.imwrite('./output/detected.jpg', detected)
	if len(bbox)>1:
		issues.append("More than one face detected - Total number = "+str(len(bbox)))
		bbox = bbox[0]
	elif len(bbox)==1:
		bbox = bbox[0]
	else:
		issues.append("No faces detected")
		out = jsonable_encoder({"output":[{'issues':issues},{'score':[1]},{'name':['No face score']}]})
		return out
	
	x,y,x1,y1 = bbox
	img = original[y:y1, x:x1, :]
	
	sobel, blur, blurScore = check_blur(img)
	if save:
		cv2.imwrite('./output/sobel_edge.jpg', sobel)
	if blur:
		issues.append("Face is Blurred")
		score.append(float(1-float(blurScore)))
		name.append("blurred image score")
	else:
		score.append(float(blurScore))
		name.append("Not blurred image score")
	resized, small = check_dimensions(original, img)
	if save:
		cv2.imwrite('./output/resized.jpg', resized)
	if small:
		issues.append("Too small")

	fake,score_realFake = check_real(resized[:,:,:])
	if fake:
		issues.append("Fake image")
		name.append("fake image score")
		score.append(float(score_realFake))
	else:
		name.append("real image score")
		score.append(float(score_realFake))

	obstruct,score_Obstruct = check_obstruct(resized[:,:,:])
	if obstruct:
		issues.append("Obstructed face")
		name.append("obstructed face detection")
		score.append(float(score_Obstruct))
	else:
		name.append("No obstruction detection")
		score.append(float(score_Obstruct))
	
	lowLight,score_lowLight = low_light_detection(original)
	if lowLight:
		issues.append("low light")
		name.append("low light detection")
		score.append(float(score_Obstruct))
	else:
		name.append("normal lighting")
		score.append(float(score_Obstruct))
	
	
	end = time.time()
	# print("time : ",end-start)
	out = jsonable_encoder({"output":[{'issues':issues},{'score':score},{"name":name}]})
	return out

if __name__ == '__main__':
	path = './images/'
	for img_path in [path+i for i in sorted(os.listdir(path))]:
		ML(img_path)