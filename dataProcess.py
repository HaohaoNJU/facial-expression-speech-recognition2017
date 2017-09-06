# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:44:05 2017

@author: 浩浩
"""

# dataProcess
#通过测试
import numpy as np
from cv2 import VideoCapture
import cv2
import skimage
import random
from pydub import AudioSegment
import subprocess
import arff
import os, sys
import pickle
import csv
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from sklearn import preprocessing
import scipy.io.wavfile as wav
def equalizeImgList(imgList, row = 100, col = 100):
	'''
	Given a frameList, normalize (resize) all the frames to a common size
	'''
	# OPTIMIZATION POSSIBLE

	newFrameList = []

	for i in range(imgList.shape[0]):
		try:
			newImg = cv2.resize(imgList[i], (row, col))
		except:
			print ('Resizing error found for image with dimensions', imgList[i].shape)
			continue

		newImg = np.array(newImg, dtype = np.uint8)
		newFrameList.append(newImg)

	newFrameList = np.array(newFrameList)
	return newFrameList

def getTruthVal(fileName):
	'''
	Take the fileName (CSV File) of the truth file
	Returns a dictionary which contains mapping between fileName and 5 attitude scores
	'''
    #已经测试通过可以放心使用
	trueMap = {}
	trueVal = np.genfromtxt(fileName, delimiter=',', dtype=None)
	# Assuming we get a csv file

	for i in range(1,trueVal.shape[0]):
		# The first one is not a valid entry
		tmpList = list(trueVal[i][1:])
		tmpList = [float(x) for x in tmpList]
		tmpKey = trueVal[i][0][:-4]
		trueMap[tmpKey] = np.array(tmpList)

	return trueMap

def readFromFileAudioFetA(fileName, feature, clusterSize = 4):
	filePath = 'tmpData/audioFetA/'
	fileName = filePath+fileName+'.p'
	if (not os.path.isfile(fileName)):
		return []
	fetList = pickle.load(open(fileName, 'rb'))
	tmpList = []
	for i in range(len(fetList)):
		audioFet = np.array(fetList[i]['data'][0][1:-1])
		# The first and last values are unnecessary
		tmpList.append(audioFet)

	if (len(tmpList) == 0):
		return []

	tmpFeature = []
	if ('minmax' in feature):
		newList = []
		if (clusterSize <= 0):
			tmpFeature = np.max(tmpList, axis = 0)
			tmpFeature = np.append(tmpFeature, np.min(tmpList, axis = 0))
			newList.append(tmpFeature)
		else:
			for i in range(len(tmpList) - clusterSize + 1):
				clusterList = np.array(tmpList[i:i+clusterSize])
				tmpFeature = np.max(clusterList, axis = 0)
				# print tmpFeature.shape
				tmpFeature = np.append(tmpFeature, np.min(clusterList, axis = 0))
				# print tmpFeature.shape
				newList.append(tmpFeature)
		tmpList = newList
	elif ('avg' in feature):
		newList = []
		if (clusterSize <= 0):
			newList.append(np.mean(tmpList, axis = 0))
		else:
			for i in range(len(tmpList) - clusterSize + 1):
				clusterList = np.array(tmpList[i:i+clusterSize])
				tmpFeature = np.mean(clusterList, axis = 0)
				newList.append(tmpFeature)
		tmpList = newList

	# tmpList = np.array(tmpList)
	# print tmpList.shape
	return tmpList

def readFromFileFetA(fileName, skipLength = 2, augment = False):
	filePath = 'tmpData/visualFetA/'
	fileName = filePath+fileName+'.npy'
	if (not os.path.isfile(fileName)):
		return []
	newImgList = np.load(fileName)
	tmpList = []
	row, col = 50, 50
	startInd = random.randint(0, skipLength-1)
	for i in range(startInd, newImgList.shape[0], skipLength):
		newImg = cv2.resize(newImgList[i], (row, col))
		if augment:
			if (random.randint(1,2) > 1):
				newImg = np.fliplr(newImg)
		tmpList.append(newImg)
	tmpList = np.array(tmpList)
	return tmpList
#feature B看起来是预先保存在.npy文件里面的数据
def readFromFileFetB(fileName,skipLength = 2):
	filePath = 'E:/Video_for_test/fetB/'
	fileName = filePath+fileName+'.npy'
	if (not os.path.isfile(fileName)):
		return []
	fetList = np.load(fileName)
	tmpList = []
    #生成随机整数
	startInd = random.randint(0, skipLength-1)
	for i in range(startInd, fetList.shape[0], skipLength):
		tmpList.append(fetList[i])
	tmpList = np.array(tmpList)
	tmpList.astype(np.float32)
	return tmpList
def readFromFileFetBT(fileName,skipLength = 2):
	filePath = 'E:/Video_as_test/fetB/'
	fileName = filePath+fileName+'.npy'
	if (not os.path.isfile(fileName)):
		return []
	fetList = np.load(fileName)
	tmpList = []
    #生成随机整数
	startInd = random.randint(0, skipLength-1)
	for i in range(startInd, fetList.shape[0], skipLength):
		tmpList.append(fetList[i])
	tmpList = np.array(tmpList)
	tmpList.astype(np.float32)
	return tmpList

def readFromFileFetC(fileName, skipLength = 2, augment = False):
	filePath = 'E:/Video_for_test/fetC/'
	fileName = filePath+fileName+'.npy'
	if (not os.path.isfile(fileName)):
		return []
	newImgList = np.load(fileName)
	tmpList = []
	row, col = 50, 50
	startInd = random.randint(0, skipLength-1)
	for i in range(startInd, newImgList.shape[0], skipLength):
		ind1 = random.randint(0, 8)
		ind2 = random.randint(0, 8)
		l = 100 - (ind1 + ind2)
		newImg = newImgList[i][ind1:ind1 + l, ind2:ind2 + l]
		newImg = cv2.resize(newImg, (row, col))
		if augment:
			if (random.randint(1,2) > 1):
				newImg = np.fliplr(newImg)
		tmpList.append(newImg)
	tmpList = np.array(tmpList)
	return tmpList
def readFromFileFetCT(fileName, skipLength = 2, augment = False):
	filePath = 'E:/Video_as_test/fetC/'
	fileName = filePath+fileName+'.npy'
	if (not os.path.isfile(fileName)):
		return []
	newImgList = np.load(fileName)
	tmpList = []
	row, col = 50, 50
	startInd = random.randint(0, skipLength-1)
	for i in range(startInd, newImgList.shape[0], skipLength):
		ind1 = random.randint(0, 8)
		ind2 = random.randint(0, 8)
		l = 100 - (ind1 + ind2)
		newImg = newImgList[i][ind1:ind1 + l, ind2:ind2 + l]
		newImg = cv2.resize(newImg, (row, col))
		if augment:
			if (random.randint(1,2) > 1):
				newImg = np.fliplr(newImg)
		tmpList.append(newImg)
	tmpList = np.array(tmpList)
	return tmpList


def readFromFileFetF(fileName, poolType = 'max', numSamples = 5, numTotSamples = None, numPools = 10, randomFlag = False):
	filePath = 'E:/Video_for_test/fetF/'
	fileName = filePath+fileName+'.npy'
	if (not os.path.isfile(fileName)):
		return []

	fetList = np.load(fileName)
	if (fetList.shape[1] == 0):
		return []

	newFetList = np.empty((0, fetList.shape[2]))
	totCnt = 0

	startInd = 0
	if (randomFlag):
		startInd = random.randint(0, fetList.shape[0] - 1)

	for j in range(startInd, fetList.shape[0]):
		for k in range(numSamples):

			tmpList = []
			for i in range(numPools):
				tmpList.append(random.randint(0, fetList[j].shape[0] - 1))

			tmpFetList = fetList[j].take(tmpList, axis = 0)

			if (poolType == 'max'):
				tmpFetList = np.max(tmpFetList, axis = 0).reshape(1,-1)
			elif (poolType == 'avg'):
				tmpFetList = np.mean(tmpFetList, axis = 0).reshape(1,-1)

			# print tmpFetList.shape, newFetList.shape
			newFetList = np.concatenate((newFetList, tmpFetList), axis = 0)
			totCnt += 1
			if (numTotSamples is not None):
				if (totCnt >= numTotSamples):
					print ('fetF_train特征的形状是：',newFetList.shape)
					return newFetList
                
	print ('fetF_train特征的形状是：',newFetList.shape)	
	return newFetList
def readFromFileFetFT(fileName, poolType = 'max', numSamples = 5, numTotSamples = None, numPools = 10, randomFlag = False):
	filePath = 'E:/Video_as_test/fetF/'
	fileName = filePath+fileName+'.npy'
	if (not os.path.isfile(fileName)):
		return []

	fetList = np.load(fileName)
	if (fetList.shape[1] == 0):
		return []

	newFetList = np.empty((0, fetList.shape[2]))
	totCnt = 0

	startInd = 0
	if (randomFlag):
		startInd = random.randint(0, fetList.shape[0] - 1)

	for j in range(startInd, fetList.shape[0]):
		for k in range(numSamples):

			tmpList = []
			for i in range(numPools):
				tmpList.append(random.randint(0, fetList[j].shape[0] - 1))

			tmpFetList = fetList[j].take(tmpList, axis = 0)

			if (poolType == 'max'):
				tmpFetList = np.max(tmpFetList, axis = 0).reshape(1,-1)
			elif (poolType == 'avg'):
				tmpFetList = np.mean(tmpFetList, axis = 0).reshape(1,-1)

			# print tmpFetList.shape, newFetList.shape
			newFetList = np.concatenate((newFetList, tmpFetList), axis = 0)
			totCnt += 1
			if (numTotSamples is not None):
				if (totCnt >= numTotSamples):
					print ('fetF_test特征的形状是：',newFetList.shape)
					return newFetList
                
	print ('fetF_test特征的形状是：',newFetList.shape)	
	return newFetList
'''
############下下下###############READ_DATA####################################
'''
def readData(fileNames, trueVal = None, feature = 'A', poolType = 'avg', printFlag = False, clusterSize = 4):
	X = []
	Y = []
	X1 = []
	X2 = []
	# CAN BE OPTIMIZED
	for i in range(len(fileNames)):
		fileName = fileNames[i]
		if (feature == 'A'):
			imgList = readFromFileFetA(fileName, 6, augment = False)
		elif (feature == 'B'):
			imgList = readFromFileFetB(fileName, 2)
		elif (feature == 'BT'):
			imgList = readFromFileFetBT(fileName, 2)			
		elif (feature == 'C'):
			imgList = readFromFileFetC(fileName, 5, augment = True)
		elif (feature == 'CT'):
			imgList = readFromFileFetCT(fileName, 5, augment = True)
		elif (feature == 'F'):
			imgList = readFromFileFetF(fileName, poolType = poolType, numSamples = 20, numPools = 10)
		elif (feature == 'FT'):
			imgList = readFromFileFetFT(fileName, poolType = poolType, numSamples = 20, numPools = 10)
		elif ('AudioA' in feature):
			imgList = readFromFileAudioFetA(fileName, feature, clusterSize = clusterSize)
		elif (feature == 'CF'):
			imgList = readFromFileFetC(fileName, 5, augment = True)
			vggList = readFromFileFetF(fileName, poolType = poolType, numSamples = len(imgList), numTotSamples = len(imgList), numPools = 10, randomFlag = True)
			if (len(vggList) == 0):
				continue
		elif (feature == 'CFT'):
			imgList = readFromFileFetCT(fileName, 5, augment = True)
			vggList = readFromFileFetFT(fileName, poolType = poolType, numSamples = len(imgList), numTotSamples = len(imgList), numPools = 10, randomFlag = True)
			if (len(vggList) == 0):
				continue
		if (len(imgList) == 0):
			print('没有读出来frames!')
			continue
#############################

		if (trueVal is not None):
			#Y.extend([trueVal[fileName.encode(encoding='utf-8')]]*len(imgList))
			try:
				Y.extend([trueVal[fileName.encode(encoding='utf-8')]]*len(imgList))
				if (printFlag):
					print ('\r', (i*(1.0))/len(fileNames), 'part reading completed')
					sys.stdout.flush()
			except:
				print("这个视频文件没有读出来，没有标签")
				continue
		if (feature == 'CF'):
			X1.extend(imgList)
			X2.extend(vggList)
		elif(feature == 'CFT'):
			X1.extend(imgList)
			X2.extend(vggList)
		else:
			X.extend(imgList)
#################################################
	Y=np.array(Y, dtype = np.float16)
	if (feature == 'CF'):
		X1 = np.array(X1, dtype = np.float16)
		X2 = np.array(X2, dtype = np.float16)
		# print X1.shape, X2.shape, Y.shape
		return X1, X2, Y
	elif (feature == 'CFT'):
		X1 = np.array(X1, dtype = np.float16)
		X2 = np.array(X2, dtype = np.float16)
		# print X1.shape, X2.shape, Y.shape
		return X1, X2, Y
	else:
		# print len(X)
		# print len(X[0])
		X = np.array(X, dtype = np.float16)
		return X, Y
	# print X.shape, Y.shape
	return X, Y
'''
######上上上##################READ_DATA##################上上上##################
'''
def getAudioFeatureAList(fileName, segLen = 4, overlap = 3):
	'''
	Get IS10 features from an audio file (In WAV format)
	Assumes already present in the opensmile directory
	Returns a list of dictionaries containing the features for the segments
	'''
	fetList = []
	song = AudioSegment.from_wav(fileName)
	segLen = segLen*1000
	overlap = overlap*1000
	# Things done in ms
	totLen = len(song)
	for i in range(0, totLen - segLen, segLen - overlap):
		tmpSong = song[i:i+segLen]
		tmpSong.export("E:/Video_for_test/audioFetA/tmpAudioFeatureA.wav", format="wav")
		p = subprocess.Popen('./SMILExtract -l 1 -C config/IS10_paraling.conf -I tmpAudioFeatureA.wav -O tmpFeature.arff', shell=True, stdout=subprocess.PIPE)
		p.wait()
		data = arff.load(open('tmpFeature.arff', 'rb'))
		fetList.append(data)
		# Removing the file for the next one
		p = subprocess.Popen('rm tmpFeature.arff', shell=True, stdout=subprocess.PIPE)
		p.wait()
	return fetList
def randomCrops(frame, numCrops = 8, row = 224, col = 224):
	X = []
	rows = frame.shape[0]
	cols = frame.shape[1]

	if (rows*cols == 0):
		return np.array([])

	if ((rows < row) or (cols < col)):
		return np.array([])

	meanVal = np.array([103.939, 116.779, 123.68])   # BGR
	for i in range(numCrops):
		x = random.randint(0, rows - row)
		y = random.randint(0, cols - col)
		img = frame[x:(x+row), y:(y+col), :].astype(np.float32)
		img = img[::-1]  # switch to BGR, since VGG requires BGR images
		img -= meanVal
		X.append(img)

	X = np.array(X)
	return X

def GetBGFeatures(frameList, model, numCrops = 8):
	fetList = []
	row, col = 224, 224

	# print len(frameList)
	for i in range(len(frameList)):
		frame = frameList[i]
		X = randomCrops(frame, numCrops, row = row, col = col)
		X = X.reshape(X.shape[0], row, col,3)

		embed = model.predict(X)
		# Don't do max pool right now. Store all the embeddings to perform augmentation later.
		# finX = np.max(embed, axis = 0)
		fetList.append(embed)

	fetList = np.array(fetList)
	# print fetList.shape
	return fetList

def evaluateTraits(p, gt, printFlag = True):
	# Currently allowing negative values

	if (len(p) == len(gt)):
		for i in range(len(p)):
			if (len(p[i]) != 5) or (len(gt[i]) != 5):
				print ("Inputs must be a list of 5 values within the range [0,1]. Traits could not be evaluated.")
				# return
			for j in range(len(p[i])):
				if p[i][j] < 0 or p[i][j] > 1 or gt[i][j] < 0 or gt[i][j] > 1:
					if printFlag:
						print ("Inputs must be values in the range [0,1]. Traits could not be evaluated.")
						# return
	
	errors = np.abs(p-gt)
	meanAccs = 1-np.mean(errors, axis=0)

	if printFlag:
		print ("\nAverage accuracy of "+str(np.mean(meanAccs))+": ")
		
		# These scores are reported.
		print ("Accuracy predicting Extraversion: "+str(meanAccs[0]))
		print ("Accuracy predicting Agreeableness: "+str(meanAccs[1]))
		print ("Accuracy predicting Conscientiousness: "+str(meanAccs[2]))
		print ("Accuracy predicting Neuroticism: "+str(meanAccs[3]))
		print ("Accuracy predicting Openness to Experience: "+str(meanAccs[4]))
		print ("\n")
		
	meanAccs = np.mean(meanAccs)

	# if (not printFlag):
	# 	print meanAccs

	return meanAccs

def generatePredFile(p, subset='validation'):
	vnames = []
	with open('../training/'+subset+'_gt.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		next(reader, None)
		for row in reader:
			vnames.append(row[0])
	csvfile.close()
	with open('tmpData/predictions/predictions.csv', 'wb') as csvfile:
		gtwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		gtwriter.writerow(['VideoName', 'ValueExtraversion', 'ValueAgreeableness', 'ValueConscientiousness', 'ValueNeurotisicm','ValueOpenness'])
		for i in range(0,len(vnames)):
			vnames[i] = vnames[i].strip('.mp4')
			if (vnames[i] not in p):
				p[vnames[i]] = [0.5]*5
			if (isinstance(p[vnames[i]], np.float64)):
				p[vnames[i]] = [0.5]*5
			# print vnames[i]
			# print p[vnames[i]]
			if (len(p[vnames[i]]) == 1):
				p[vnames[i]] = p[vnames[i]][0]
			gtwriter.writerow([vnames[i]+'.mp4', p[vnames[i]][0], p[vnames[i]][1], p[vnames[i]][2], p[vnames[i]][3], p[vnames[i]][4]])
	csvfile.close()
def getAudiofet(audioPath=None,fileName=None,audNames=None):
    trueVal=getTruthVal(fileName)
    lis=list(trueVal.keys())
    X=[]
    Y=[]
    for i in audNames:
        if(i.encode(encoding='utf-8') in lis and os.path.exists(audioPath+i+'.wav')):
            fs,audio=wav.read(audioPath+i+'.wav',mmap=True)
            aaa=mfcc(audio,samplerate=fs)
            bbb=logfbank(audio,samplerate=fs)
            ccc=np.concatenate((aaa,bbb),axis=1)
            if(ccc.shape[0] < 3064):
                zeros=np.zeros((int(3064-ccc.shape[0]),39))
                ccc=np.concatenate((ccc,zeros),axis=0)
            elif(ccc.shape[0] > 3064):
                ccc=ccc[:3064,:]
            ccc1=delta(ccc,1)
            ccc2=delta(ccc,2)
            aaa=np.concatenate((ccc,ccc1,ccc2),axis=1)
            aaa=np.transpose(aaa)
#            aaa=preprocessing.scale(aaa)
            X.append(aaa)
            Y.extend([trueVal[i.encode(encoding='utf-8')]])
        else:
            continue
    X=np.array(X,dtype=np.float32)
    Y=np.array(Y,dtype=np.float32)
#按着列数归一化处理

    return X,Y
def getAudiofet2(audioPath=None,fileName=None,audNames=None):
    trueVal=getTruthVal(fileName)
    lis=list(trueVal.keys())
    X=[]
    Y=[]
    for i in audNames:
        if(i.encode(encoding='utf-8') in lis and os.path.exists(audioPath+i+'.wav')):
            fs,audio=wav.read(audioPath+i+'.wav',mmap=True)
            aaa=mfcc(audio,samplerate=fs)
            bbb=logfbank(audio,samplerate=fs)
            ccc=np.concatenate((aaa,bbb),axis=1)
            ccc1=delta(ccc,1)
            ccc2=delta(ccc,2)
            aaa=np.concatenate((ccc,ccc1,ccc2),axis=1)
            aaa=np.mean(aaa,axis=0)
            aaa=preprocessing.scale(aaa)
            X.append(aaa)
            Y.extend([trueVal[i.encode(encoding='utf-8')]])
        else:
            continue
    X=np.array(X,dtype=np.float32)
    Y=np.array(Y,dtype=np.float32)
#按着列数归一化处理
 
    return X,Y