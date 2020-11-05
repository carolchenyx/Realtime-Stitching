# USAGE
# python realtime_stitching.py

# import the necessary packages
from __future__ import print_function
from pyimagesearch.basicmotiondetector import BasicMotionDetector
from pyimagesearch.panorama import Stitcher
from imutils.video import FileVideoStream
import numpy as np
import datetime
import imutils
import time
import cv2
import os

os.makedirs("img/res",exist_ok=True)
os.makedirs("img/left",exist_ok=True)
os.makedirs("img/right",exist_ok=True)
# initialize the video streams and allow them to warmup
print("[INFO] starting cameras...")
leftStream = FileVideoStream("test/0_flip.mp4").start()
rightStream = FileVideoStream("test/1_flip.mp4").start()
# leftStream ="test/0.mp4"
# rightStream = "test/1.mp4"
time.sleep(2.0)

# initialize the image stitcher, motion detector, and total
# number of frames read
stitcher = Stitcher()
# motion = BasicMotionDetector(minArea=500)
total = 0

mtx = np.array(
	[[658.9459183818934, 0.0, 302.07862140526515], [0.0, 618.1322354192894, 118.56744732777575], [0.0, 0.0, 1.0]])
dist = np.array(
	[[-0.69878120107968], [-0.8719865895006724], [0.12354602539201477], [-0.005475419126478167], [2.5004260496221598]])
K = np.array(
	[[277.01115264409475, 0.0, 305.257737360851], [0.0, 274.1672538344219, 223.4453462870637], [0.0, 0.0, 1.0]])
D = np.array([[0.2197595338207982], [-0.0656242869051858], [-0.052999115743155044], [0.14021913903495045]])

def cut_image(img, bottom=0, top=0, left=0, right=0):
	height, width = img.shape[0], img.shape[1]
	return np.asarray(img[top: height - bottom, left: width - right])

# loop over frames from the video streams
while True:
	# grab the frames from their respective video streams
	left = leftStream.read()
	right = rightStream.read()
	cv2.imshow("left", cv2.resize(left,(640,480)))
	cv2.imshow("right", cv2.resize(right,(640,480)))
	widthl, heightl = int(left.shape[1]), int(left.shape[0])
	P_l = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (widthl, heightl), None)
	mapx2_l, mapy2_l = cv2.fisheye.initUndistortRectifyMap(K, D, None, P_l, (widthl, heightl), cv2.CV_32F)
	widthr, heightr = int(right.shape[1]), int(right.shape[0])
	P_r = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (widthr, heightr), None)
	mapx2_r, mapy2_r = cv2.fisheye.initUndistortRectifyMap(K, D, None, P_r, (widthr, heightr), cv2.CV_32F)

	left = cv2.remap(left, mapx2_l, mapy2_l, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
	right = cv2.remap(right, mapx2_r, mapy2_r, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
	# left = cut_image(left, bottom=100, top=100, left=100, right=100)
	# right = cut_image(right, bottom=100, top=100, left=100, right=100)




	# resize the frames
	left = imutils.resize(left, width=800)
	right = imutils.resize(right, width=800)

	# stitch the frames together to form the panorama
	# IMPORTANT: you might have to change this line of code
	# depending on how your cameras are oriented; frames
	# should be supplied in left-to-right order
	result = stitcher.stitch([left, right])

	# no homograpy could be computed
	if result is None:
		print("[INFO] homography could not be computed")
		break

	# convert the panorama to grayscale, blur it slightly, update
	# the motion detector
	# gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
	# gray = cv2.GaussianBlur(gray, (21, 21), 0)
	# locs = motion.update(gray)

	# only process the panorama for motion if a nice average has
	# been built up
	# if total > 32 and len(locs) > 0:
	# 	# initialize the minimum and maximum (x, y)-coordinates,
	# 	# respectively
	# 	(minX, minY) = (np.inf, np.inf)
	# 	(maxX, maxY) = (-np.inf, -np.inf)
	#
	# 	# loop over the locations of motion and accumulate the
	# 	# minimum and maximum locations of the bounding boxes
	# 	for l in locs:
	# 		(x, y, w, h) = cv2.boundingRect(l)
	# 		(minX, maxX) = (min(minX, x), max(maxX, x + w))
	# 		(minY, maxY) = (min(minY, y), max(maxY, y + h))
	#
	# 	# draw the bounding box
	# 	cv2.rectangle(result, (minX, minY), (maxX, maxY),
	# 		(0, 0, 255), 3)

	# increment the total number of frames read and draw the
	# timestamp on the image
	total += 1
	timestamp = datetime.datetime.now()
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	cv2.putText(result, ts, (10, result.shape[0] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)



	# show the output images
	cv2.imshow("result",result)
	# cv2.imshow("left", left)
	# cv2.imshow("right", right)
	cv2.imwrite("img/res/Result_{}.jpg".format(total), result)
	cv2.imwrite("img/left/Left Frame_{}.jpg".format(total), left)
	cv2.imwrite("img/right/Right Frame_{}.jpg".format(total), right)
	key = cv2.waitKey(250) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# do a bit of cleanup
	print("[INFO] cleaning up...")
cv2.destroyAllWindows()
	# leftStream.stop()
	# rightStream.stop()