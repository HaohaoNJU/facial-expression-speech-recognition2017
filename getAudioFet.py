# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:48:35 2017

@author: 浩浩
"""
#getAudioFet.py
#从剪辑中提取音频功能。需要输入要保存和读取的目录。
from dataProcess import *
import pickle
import os
import sys
def getAudioFetA():
	'''
	Extracting audio features using the interspeech10 config
	Each audio track is broken into multiple overlapping segments, for which the features are computed
	The generated file for each video is a csv
	'''
	# fileName = '../training/training_gt.csv'
	# trueMap = getTruthVal(fileName)
	print( 'Started extracting audio features')
	videoPath = 'E:/Video_for_test/'
	vidNames = os.listdir(videoPath)
	vidNames = [x for x in vidNames if x.endswith(".mp4")]
	openSmilePath = 'F:/opensmile-2.3.0/'
	aaa='test/test.wav'
	# vidNames.extend(vidNamesTest)
	# Initialize detectors, load it for face detection
	saveFetPath = 'E:/Video_for_test/audioFetA/'
	if not os.path.exists(saveFetPath):
	    os.makedirs(saveFetPath)
	vidNames = vidNames
#改变当前工作路径
	os.chdir(openSmilePath)
	for i in range(1):
		fileName = vidNames[i]
		if (os.path.isfile(saveFetPath+fileName.strip('.mp4.wav')+'.p')):
			continue
		fetList = getAudioFeatureAList(videoPath+aaa, segLen = 4, overlap = 3)
		#savePath = saveFetPath + fileName.strip('.mp4')
		savePath=saveFetPath + 'test'
		pickle.dump(fetList, open(savePath+'.p', 'wb'))
		print('\r', (i*(1.0))/len(vidNames), 'part completed. Currently at file:', fileName)
		sys.stdout.flush()
	print('\n')
if __name__ == "__main__":
	getAudioFetA()
