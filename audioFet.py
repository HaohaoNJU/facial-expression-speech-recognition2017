# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:32:23 2017

@author: 浩浩
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD 
from keras.utils import np_utils, generic_utils
import keras.layers.normalization
from keras import backend as tf
import numpy as np
from readVideo import *
import random
from sklearn import preprocessing
from sklearn.externals import joblib
from python_speech_features import mfcc
from python_speech_features import delta
import scipy.io.wavfile as wav
import csv
import os

#keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
audioPath = 'E:/Video_for_test/test/'
audioPathTest= r'E:/Video_as_test/test/'
audNames = os.listdir(audioPath)
audNames = [x for x in audNames if x.endswith('.wav')]
audNamesTest=os.listdir(audioPathTest)
audNamesTest=[x for x in audNamesTest if x.endswith('.wav')]
fileName = r'E:/Video_for_test/training_gt.csv'
trueVal=getTruthVal(fileName)
lis=list(trueVal.keys())
for i in range(len(audNames)):
	audNames[i] = audNames[i][:-4]
for i in range(len(audNamesTest)):
	audNamesTest[i] = audNamesTest[i][:-4]
###shape=3064*13
#X_train,Y_train=getAudiofet(audioPath,fileName,audNames)
X_test,Y_test=getAudiofet(audioPathTest,fileName,audNamesTest)
#X_train=X_train.reshape((X_train.shape[0],3064,117,1))
X_test=X_test.reshape((X_test.shape[0],3064,117,1))
###可以读出X,Y,test,train了
numPerBatch = 1
numBatch = (len(audNames)/numPerBatch)
#每隔多久存一次模型的权重，训练多的时候可以用
model_save_interval = 1
num_epochs = 100

model_file_name = 'E:/Video_for_test/models/audA/audioFetA_Conv_CTC'

model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# 10卷积层，128 for first 4 layers，256 for the last 6 Conv layers
#1024 units for 3 full-connected layers
#pool-size=3*1 & filter-size=3*5

model.add(Convolution2D(128, 5, 3, border_mode='valid', input_shape=(3064,117,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 3)))
model.add(Convolution2D(128, 5, 3))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(128, 5, 3))
#model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(128, 5, 3))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Convolution2D(256, 5, 3))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(256,5, 3))
#model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(256,5, 3))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(256, 5, 3))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(256, 5, 3))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(256, 5, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.3))
model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256,kernel_initializer='random_uniform',
#                keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None),\
                bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(5))
model.add(Activation('sigmoid'))
##优化器
sgd=SGD(lr=1e-4,momentum=0.9,decay=1e-6,nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
# Prefer mean_absolute_error, since that is the one used for the challenge

# Saving model
jsonString = model.to_json()
open(model_file_name + '.json', 'w').write(jsonString)

minScore = 1.0

# raw_input('WAIT')

print ('Training started...')
for k in range(int(num_epochs)):
	#shuffle the data points before going through them
	random.shuffle(audNames)
	progbar = generic_utils.Progbar(len(audNames))

	for i in range(int(numBatch)):
		# Read numPerBatch files, get the images and answers

		# print 'Starting reading the batch'
		X_batch, Y_batch = getAudiofet(audioPath,fileName,audNames[i:i+1])
		# print X_batch.shape
		X_batch = X_batch.reshape(X_batch.shape[0], 3064, 117, 1)
		Y_batch = getAudiofet2(audioPath,fileName,audNames[i*numPerBatch:(1+i)*numPerBatch])
		print ('Training on Batch')
		try:
			loss = model.train_on_batch(X_batch, Y_batch)
		# Train the model
		# print 'Finished training on Batch'

			progbar.add(numPerBatch, values=[("train loss", loss)])
		except:
			continue
	#print type(loss)
#	if k%model_save_interval == 0:
#		model.save_weights(model_file_name + '_epoch_{}'.format(k))
	score = model.evaluate(X_test, Y_test, verbose=0)
	print ("For epoch", k, ",Testing loss is", score)
	if minScore > score:
		model.save_weights(model_file_name + '.hdf5', overwrite = True)
		minScore = score
