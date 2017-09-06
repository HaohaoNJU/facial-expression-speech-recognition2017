# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:35:40 2017
令人激动的第一次fusion！！！
@author: 浩浩
"""

#vggFetCConcat.py
#用于合并面部和基于BG的功能的网络。
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Activation, Flatten, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.models import model_from_json
import pickle
import numpy as np 
from readVideo import *
import random

from sklearn import preprocessing
from sklearn.externals import joblib

def vggFetF():
	model_file_name = 'E:/Video_for_test/models/fetF/visualFetF_VGGExtSamples_5_128_4096_avg'

	model = Sequential()
	# # input: 4096 dimension vectors.
	model.add(Dense(128, input_dim = 4096, init='uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Dense(5))
	model.add(Activation('sigmoid'))

	print( model_file_name)

	model.load_weights(model_file_name + '.hdf5')

	#Remove the last layers to get the 128D representations
	model.layers.pop()
	model.layers.pop()

	# print model.layers
	# Freeze model
	for i in range(len(model.layers)):
	    model.layers[i].trainable = False

	# Fix required for the output to be 4096D in newer versions of keras
	# model.outputs = [model.layers[-1].output]
	# model.layers[-1].outbound_nodes = []

	print ('vggFetF Model loading complete!')

	return model

def convFetC():
	model_file_name = 'E:/Video_for_test/models/fetC/32_64_256/visualFetC_Conv_Augmented_32_64_256'

	print (model_file_name)

	newModel = model_from_json(open(model_file_name + '.json').read())
	newModel.load_weights(model_file_name + '.hdf5')

	#Remove the last layers to get the 256D representations
	newModel.layers.pop()
	newModel.layers.pop()

	# print newModel.layers
	# Freeze model
	for i in range(len(newModel.layers)):
	    newModel.layers[i].trainable = False

	# Fix required for the output to be 4096D in newer versions of keras
	# newModel.outputs = [newModel.layers[-1].output]
	# newModel.layers[-1].outbound_nodes = []

	print ('convFetC Model loading complete!')

	return newModel

poolType = 'avg'
row, col = 50, 50
splitVal = 0.9
############################################
videoPath = r'E:/Video_for_test/'
videoPathTest= r'E:/Video_as_test/'
vidNames = os.listdir(videoPath)
vidNames = [x for x in vidNames if x.endswith(".mp4")]
vidNamesTest=os.listdir(videoPathTest)
vidNamesTest=[x for x in vidNamesTest if x.endswith('.mp4')]
fileName = r'E:/Video_for_test/training_gt.csv'
trueVal = getTruthVal(fileName)
for i in range(len(vidNames)):
	vidNames[i] = vidNames[i].strip('.mp4')
for i in range(len(vidNamesTest)):
	vidNamesTest[i] = vidNamesTest[i].strip('.mp4')
row,col=50,50
#splitVal = 0.9
#vidNamesTest = vidNames[int(splitVal*len(vidNames))+1:]
#vidNames = vidNames[:int(splitVal*len(vidNames))]
#X代表特征矩阵，Y代表标签
#每一个视频中，X的特征矩阵数量和Y的标签都是一样的
############################################
X_test_conv, X_test_vgg, Y_test = readData(vidNamesTest, trueVal, 'CFT', poolType = poolType)
X_test_conv = X_test_conv.reshape(X_test_conv.shape[0],row, col,1)
X_test_conv = X_test_conv.astype('float32')
X_test_conv /= 255

numPerBatch = 1
numBatch = (len(vidNames)/numPerBatch)

model_save_interval = 1
num_epochs = 100

model_file_name = 'E:/Video_for_test/models/fetConcat/VGG_FetC_Concat_(256_128)' + poolType

vggModel = vggFetF()
convModel = convFetC()

# print vggModel.input_shape
# print vggModel.output_shape
# print convModel.input_shape
# print convModel.output_shape

imgInput = Input(shape=(row, col,1), name = 'imgInput')
imgRep = convModel(imgInput)
imgRep.trainable = False

vggInput = Input(shape=(4096,), name = 'vggInput')
vggRep = vggModel(vggInput)
vggRep.trainable = False

merged = merge([vggRep,imgRep], mode='concat')
hidDrop = Dropout(0.25)(merged)

# Can add another hidden layer in between
# Don't add any, causes over fitting. Maybe a 32 node layer would be fine
# hidLayer = Dense(32, activation ='relu', name = 'hidden')(hidDrop)
# final = Dropout(0.25)(hidLayer)

output = Dense(5, activation ='sigmoid', name = 'output')(hidDrop)

# this is our final model:
model = Model(input=[vggInput,imgInput], output=output)

model.compile(loss = {'output': 'mean_absolute_error'}, optimizer = 'rmsprop')
# Prefer mean_absolute_error, since that is the one used for the challenge

# # Saving model
jsonString = model.to_json()
open(model_file_name + '.json', 'w').write(jsonString)

# model = model_from_json(open(model_file_name + '.json').read())
# print model_file_name
# # model.load_weights(model_file_name + '_epoch_25.hdf5')
# model.load_weights(model_file_name + '.hdf5')

minScore = 1.0

print ('Training started...')
for k in range(num_epochs):
	#shuffle the data points before going through them
	random.shuffle(vidNames)

	progbar = generic_utils.Progbar(len(vidNames))

	for i in range(int(numBatch-2)):
		# Read numPerBatch files, get the images and answers
		# print 'Starting reading the batch'
		X_batch_conv, X_batch_vgg, Y_batch = readData(vidNames[(i*numPerBatch):((i+3)*numPerBatch)], trueVal, 'CF', poolType = poolType)
		X_batch_conv = X_batch_conv.reshape(X_batch_conv.shape[0],row, col,1)
		X_batch_conv = X_batch_conv.astype('float32')
		X_batch_conv /= 255
		# print 'Finished reading the batch'
		# print 'Training on Batch'
		loss = model.train_on_batch({'vggInput':X_batch_vgg,'imgInput':X_batch_conv},{'output': Y_batch})
		# Train the model
		# print 'Finished training on Batch'

		progbar.add(numPerBatch, values=[("train loss", loss)])

	#print type(loss)
	if k%model_save_interval == 0:
		model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k))

	score = model.evaluate({'vggInput': X_test_vgg,'imgInput': X_test_conv}, {'output': Y_test}, verbose=0)

	print ("For epoch", k, ",Testing loss is", score)
	if minScore > score:
		model.save_weights(model_file_name + '.hdf5', overwrite = True)
		minScore = score 
