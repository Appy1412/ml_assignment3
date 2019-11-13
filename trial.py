# -*- coding: utf-8 -*-
# @Author: patan
# @Date:   2019-11-12 17:54:26
# @Last Modified by:   patan
# @Last Modified time: 2019-11-13 06:46:30
import numpy,idx2numpy

class neuralNet:
	avgDeltaWeight=[]
	batchSize=None
	funcName = None
	outputs = []
	rate = None
	labels = None
	def __init__(self, numLayers, numNodes, numClasses):
		self.numLayers = numLayers
		self.numNodes = numNodes
		self.mat = []
		self.bias = []
		self.outputs = []
		self.numClasses = numClasses
		for i in range(numLayers):
			if(i==0):
				temp = numpy.identity(numNodes[i])
			else:
				temp = 0.01 * numpy.random.normal(size=(numNodes[i],numNodes[i-1]))
			self.bias.append(numpy.zeros((numNodes[i],1)))
			self.mat.append(temp)
		for a in self.mat:
			print (numpy.shape(a))
		exit(0)

	def reluFunc(self,reluInput):
		# numpy.min(cl)
		# return reluInput.clip(min=0)
		return numpy.maximum(0,reluInput)

	def dreluFunc(self,reluInput):
		# reluInput[reluInput<=0]=0
		# reluInput[reluInput>0]=1
		# return reluInput
		# try:
		t=numpy.sign(self.reluFunc(reluInput))
		# except:
		# 	print(self.reluFunc(reluInput))
		# 	exit()
		# try:
		t[t==-1]=0
		# except:
		# 	print(t)
		# 	exit()
		return t

	def sigmoidFunc(self,x):
		return 1/(1 + numpy.exp(-x))

	def dsigmoidFunc(self,x):
		t=sigmoidFunc(x)
		return t*(1-t)

	def linearFunc(self,x):
		return x

	def dlinearFunc(self,x):
		return np.ones(x.shape)

	def tanhFunc(self,x):
		return numpy.tanh(x)

	def dtanhFunc(self,x):
		t=tanhFunc(x)
		return 1-t**2

	def softmaxFunc(self,x):
		# return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0)
		# print("x=",x)
		t= numpy.exp(x-numpy.max(x))
		return t/numpy.sum(t,axis=0)

	def dsoftmaxFunc(self,x):
		return x/self.numClasses

	# relu = numpy.vectorize(self.reluFunc)
	# sigmoid = numpy.vectorize(sigmoidFunc)
	# linear = numpy.vectorize(linearFunc)
	# tanh = numpy.vectorize(tanhFunc)
	# softmax = numpy.vectorize(softmaxFunc)

	# drelu = numpy.vectorize(self.dreluFunc)
	# dsigmoid = numpy.vectorize(dsigmoidFunc)
	# dlinear = numpy.vectorize(dlinearFunc)
	# dtanh = numpy.vectorize(dtanhFunc)
	# dsoftmax = numpy.vectorize(dsoftmaxFunc)

	# activ = None
	# dactiv = None
	# def setActivation(self,funcName):
	# 	if(funcName=="relu"):
	# 		self.activ = self.relu
	# 		self.dactiv = self.drelu
	# 	elif(funcName=="sigmoid"):
	# 		activ = sigmoid
	# 		dactiv = dsigmoid
	# 	elif(funcName=="linear"):
	# 		activ = linear
	# 		dactiv = dlinear
	# 	elif(funcName=="tanh"):
	# 		activ = tanh
	# 		dactiv = dtanh
	# 	elif(funcName=="softmax"):
	# 		activ = softmax
	# 		dactiv = dsoftmax

	def frwdProp(self,input):
		nodesPrev = input
		self.outputs = []
		for i in range(self.numLayers):
			# print(i)
			# print(self.bias[i])
			# print(self.bias[i].shape)
			if(i==0):
				layerInput = input
			else:
				layerInput = self.outputs[-1]
			weights = self.mat[i]
			if(i==self.numLayers-1):
				self.outputs.append(self.softmaxFunc(numpy.dot(weights,layerInput)+self.bias[i]))
			elif(i==0):
				self.outputs.append(layerInput)
			elif(self.funcName=="relu"):
				self.outputs.append(self.reluFunc(numpy.dot(weights,layerInput)+self.bias[i]))

	def backProp(self,d,updateWeight):
		prevDelta = None
		# print("numLayers",self.numLayers)
		# print("outputs",len(self.outputs))
		for i in range(self.numLayers-1,0,-1):
			y = self.outputs[i]
			if(i==self.numLayers-1):
				ydash = self.dsoftmaxFunc(y)
			else:
				ydash = self.dreluFunc(y)
			# ydash = dactiv(y)
			if(i==self.numLayers-1):
				delta = (d-y)*0.1
			else:
				delta = ydash*(numpy.dot(numpy.transpose(self.mat[i+1]),prevDelta))
			prevDelta = delta
			deltaWeight = self.rate*numpy.dot(delta,numpy.transpose(self.outputs[i-1]))
			self.avgDeltaWeight[i]+=deltaWeight
			self.bias[i]+=(self.rate*delta)
			if(updateWeight):
				# print("i=",i)
				# print("matsize=",len(self.mat))
				# print("avgDeltaWeightsize=",len(self.avgDeltaWeight))
				self.mat[i]+=(self.avgDeltaWeight[i]/(self.batchSize))
				self.avgDeltaWeight[i] = numpy.zeros((self.numNodes[i],self.numNodes[i-1]))

	def getLoss(self,j):
		# print(self.outputs)
		# print(self.outputs[-1][numpy.argmax(self.outputs[-1]),0])
		x = self.outputs[-1]
		# print(x.shape)
		# val = 0
		# print("getLoss",x[j,0])
		return -1*numpy.log(x[j,0])

	def fit(self,input,labels,batchSize,epochs,rate,actFunc):
		self.rate = rate
		self.labels = labels
		self.batchSize = batchSize;
		# self.setActivation(actFunc)
		self.funcName = actFunc
		self.avgDeltaWeight=[numpy.zeros((self.numNodes[i],self.numNodes[i-1])) for i in range(1,self.numLayers)]
		self.avgDeltaWeight.insert(0,0)
		rows = input.shape[0]
		cols = input.shape[1]
		for i in range(epochs):
			totalLoss = 0
			for j in range(input.shape[0]):
				self.frwdProp((input[j,:]).reshape(cols,1))
				dtobesent = numpy.zeros((10,1))
				for k in range(10):
					if(k==labels[j,0]):
						dtobesent[k,0]=1
				if(j%batchSize==0):
					self.backProp(dtobesent,True)
				else:
					self.backProp(dtobesent,False)
				totalLoss+=self.getLoss(labels[j,0])
			print("epoch",i,", loss",totalLoss)
	def predict(self,input):
		pass
	def score(self,input,labels):
		pass
#PLAAAAAAAAAAAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGGGGGGGGGGGGGG


training_image_set = idx2numpy.convert_from_file('train-images-idx3-ubyte')
training_label_set = idx2numpy.convert_from_file('train-labels-idx1-ubyte')
test_image_set = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')
test_label_set = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')

training_y = numpy.transpose(numpy.asmatrix(training_label_set))
test_y = numpy.transpose(numpy.asmatrix(test_label_set))

training_set = numpy.zeros((training_image_set.shape[0], training_image_set.shape[1]**2))
test_set = numpy.zeros((test_image_set.shape[0], test_image_set.shape[1]**2))

for i in range(training_image_set.shape[0]):
	training_set[i, :] = training_image_set[i].flatten()

for i in range(test_image_set.shape[0]):
	test_set[i, :] = test_image_set[i].flatten()

x = training_set[:100, :]
y = training_y[:100, :]
# x = (x - x.min())/abs(x.min())

net = neuralNet(5,[784,256,128,64,10],10)
net.fit(x,y,10,100,0.1,"relu")