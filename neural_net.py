import numpy as np
from math import *

class Neural_Net:

	layers = 0
	nodes = []
	activation = ''
	learning_rate = 0.0
	weights = []
	bias = []
	delta_W = [] 


	def __init__(self, layers, nodes, activation, learning_rate):
		self.layers = layers
		self.nodes = nodes
		self.activation = activation
		self.learning_rate = learning_rate
		self.bias = np.array([[0.0 for i in range(self.nodes[j])] for j in range(layers)])
		for i in range(self.layers):
			self.weights.append([])
			self.delta_W.append([])
			if i == 0:
				continue
			for j in range(self.nodes[i]):
				w = np.array([0.01*np.random.normal(0,1) for k in range(nodes[i-1])])
				self.weights[i].append(w)
				self.delta_W[i].append([0.0 for k in range(nodes[i-1])])
		self.weights = np.array(self.weights)
		self.delta_W = np.array(self.delta_W)
		for i in range(len(self.weights)):
			self.weights[i] = np.array(self.weights[i])
			self.delta_W[i] = np.array(self.delta_W[i])

	def activation_function(self,activation, values):
		values = np.array(values)
		if activation == 'sigmoid':
			values = 1 / (1 + np.exp(-values))
			
		elif activation == 'relu':
			values[values < 0] = 0

		elif activation == 'linear':
			values = values

		elif activation == 'tanh':
			values = np.tanh(values)

		else:
			values = np.exp(values - np.max(values))
			values = values/np.sum(values, axis = 0)

		return values

	def derivative(self, activation, output):

		if activation == 'sigmoid':
			return output*(1.0-output)

		elif activation == 'linear':
			return 1

		elif activation == 'softmax':
			return np.array([0.1 for i in range(len(output))])

		elif activation == 'relu':
			d = []
			for i in range(len(output)):
				if output[i] >= 0:
					d.append(1)
				else:
					d.append(0)
			return np.array(d)

		else:
			return (1.0-output*output)


	def forwardPropagation(self, input_data):
		yi = [input_data]
		for i in range(1,self.layers):
			w = self.weights[i]
			output = np.dot(w,input_data) + self.bias[i]

			if i == self.layers-1:
				input_data = self.activation_function('softmax', output)

			else:
				input_data = self.activation_function(self.activation, output)
			yi.append(input_data)
		
		return [input_data,yi]


	def backwardPropagation(self,input_label,yi):
		num_of_classes = 10
		delta = [[] for i in range(self.layers)]
		
		di = [0.0 for i in range(num_of_classes)]
		di[input_label] = 1.0
		
		for layer in range(self.layers-1,0,-1):
			v_j = yi[layer]
			delta_j = 0.0
			if layer == self.layers-1:
				delta_j = 0.1*(di-yi[layer]) 
			else:
				delta_j = np.dot(self.weights[layer+1].T, delta[layer+1])
				delta_j *= self.derivative(self.activation, v_j)

			y_i = np.reshape(yi[layer-1], (len(yi[layer-1]),1))
			delta[layer] = delta_j
			self.bias[layer] += (self.learning_rate*delta_j)
			delta_j = np.reshape(delta_j, (len(delta_j), 1))
			delta_w = self.learning_rate*np.dot(delta_j,(y_i).T)
			self.delta_W[layer] += delta_w

	def fit(self, input_data, input_labels, batch_size, epochs):
		for epoch in range(epochs):
			error = 0.0
			cnt = 0
			for i in range(len(input_data)):
				output,yi = self.forwardPropagation(input_data[i])
				error += self.calculateError(input_labels[i][0], output)
				cnt += 1
				self.backwardPropagation(input_labels[i,0], yi)
				if cnt%batch_size == 0:
					self.delta_W /= batch_size
					self.weights += self.delta_W
					self.delta_W.fill(0)
			print ('error on epoch', epoch, 'is', error)
			
	def predict(self, input_data):
		output = []
		for i in range(len(input_data)):
			prediction = self.forwardPropagation(input_data[i])[0]
			result = np.where(prediction == np.max(prediction))
			output.append(result)
		return output

	def score(self, input_data, input_labels):
		result = self.predict(input_data)
		cnt = 0
		for i in range(len(input_data)):
			if input_labels[i,0] in result[i]:
				cnt += 1
		return cnt/len(input_data)

	def calculateError(self,input_label, output):
		error = 0.0
		for i in range(len(output)):
			if i == input_label:
				error += (-log(output[i]))
		return error







