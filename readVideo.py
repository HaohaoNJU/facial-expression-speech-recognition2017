# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 22:48:28 2017
通过主函数测试，成功保存.npy文件
@author: 浩浩
"""		
#readVideo读视频
import numpy as np
from cv2 import VideoCapture
import cv2
import os
from faceDetect import *
from dataProcess import *

def PlayVideo(fileName, redFact = 0.5):
	'''
	Plays video using opencv functions
	Press 'q' to stop in between
	returns None
	'''
	cap = VideoCapture(fileName)
	while True:
		retval, image = cap.read()
		print ( len(image), len(image[0]))
		if not retval:
			break
		image = cv2.resize(image, None, fx=redFact, fy=redFact)
		image = image[:,:,::-1]
		cv2.imshow('frame',image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

def GetFrames(fileName, redFact = 0.5, skipLength = 1, debug = False):
	'''
	returns numpy array of frames
	'''
	cap = VideoCapture(fileName)

	frameList = []
	cnt = -1

	if debug:
		print ( "Started creating Frame List")

	while True:
		retval, image = cap.read()
		cnt = (cnt+1)%skipLength
		if (cnt != 0):
			continue
		if not retval:
			break
		image = cv2.resize(image, None, fx=redFact, fy=redFact)
		image = image[:,:,::-1]
		image = np.array(image, dtype = np.uint8)
		frameList.append(image)
	cap.release()

	if debug:
		print ( "Finished creating Frame List")
	frameList = np.array(frameList)
	print("该视频总共抽取了",frameList.shape[0],"个帧文件")
	return frameList

if __name__ == "__main__":
    videoPath=r"E:/Video_for_test/"
    vidNames = os.listdir(videoPath)
    vidNames = [x for x in vidNames if x.endswith(".mp4")]
    # fileName = '8XBprf4NyOg.001.mp4'
    #PlayVideo(videoPath+fileName)
    savedVidPath = r'E:/Video_for_test/NPY/'
    savedPicPath = 'E:/Video_for_test/tmpPic.jpg'
    fileName = vidNames[0]
    frameList = None
    for i in vidNames:
        if (not os.path.isfile(videoPath + '.npy')):
            frameList = GetFrames(videoPath+i, redFact = 0.5, skipLength = 5)
            np.save(savedVidPath+i.strip('.mp4'), frameList)
        else:
            frameList = np.load(savedVidPath+'.npy')
    
    
    #print(frameList)
    #DetectFace(frameList[3])
    #DrawFace(frameList[0])
    #DetectFaceInList(frameList, None, True)
    # DetectFaceLandmarksInList(frameList, None, None)
    # DetectFaceInListDlib(frameList, None, 2, True)
    #cv2.imwrite(savedPicPath, frameList[0])
    #np.save('E:/Video_for_test/NPY/'+fileName.strip('.mp4')+'.npy',frameList)