import numpy as np

class Neural_Net:

	layers = 0
	nodes = []
	activation = ''
	learning_rate = 0.0
	weights = []
	bias = []


	def __init__(self, layers, nodes, activation, learning_rate):
		self.layers = layers
		self.nodes = nodes
		self.activation = activation
		self.learning_rate = learning_rate
		self.bias = np.array([[0 for i in range(self.nodes[j])] for j in range(layers)])
		for i in range(self.layers):
			self.weights.append([])
			if i == 0:
				continue
			for j in range(self.nodes[i]):
				w = np.array([0.01*np.random.normal(0,1) for k in range(nodes[i-1])])
				self.weights[i].append(w)
		self.weights = np.array(self.weights)
		for i in range(len(self.weights)):
			self.weights[i] = np.array(self.weights[i])


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
			values = values/values.sum()

		return values

	def derivative(self, activation, output):

		if activation == 'sigmoid':
			return output*(1.0-output)

		elif activation == 'linear':
			return 1

		elif activation == '':
			pass


	def forwardPropagation(self, input_data):
		yi = [input_data]
		for i in range(1,self.layers):
			w = self.weights[i]
			output = np.dot(w,input_data) + self.bias[i]
			yi.append(output)
			input_data = self.activation_function(self.activation, output)
		
		input_data = self.activation_function('softmax', input_data)
		return [input_data,yi]


	def backwardPropagation(self, input_label, yi):

		num_of_classes = 10
		error = [0.0 for i in range(num_of_classes)]
		delta = [[0.0 for i in range(self.nodes[j])] for j in range(self.layers)]
		

		for i in range(num_of_classes):
			if i == input_label:
				error[i] = 1.0 - yi[self.layers-1][i]
			else:
				error[i] = 0.0 - yi[self.layers-1][i]

		for layer in range(self.layers-1,0,-1):
			#print ('layer', layer)
			for node in range(self.nodes[layer]):
				#print ('node', node)
				v_j = yi[layer][node]
				delta_j = 0.0

				#output node
				if layer == self.layers-1:
					delta_j = error[node]*self.derivative(self.activation, v_j)
				
				#hidden node
				else:
					summ = 0.0
					for i in range(self.nodes[layer+1]):
						w = self.weights[layer+1][i][node]
						d = delta[layer+1][i]
						summ += w*d
					
					delta_j = summ*self.derivative(self.activation, v_j)
				for i in range(self.nodes[layer-1]):
					y_i = yi[layer-1][i]
					delta_w = delta_j*self.learning_rate*y_i
					self.weights[layer][node][i] += delta_w

				self.bias[layer][node] += delta_j*self.learning_rate


	def fit(self, input_data, input_labels, batch_size, epochs):
		for epoch in range(epochs):
			for i in range(len(input_data)):
				output,yi = self.forwardPropagation(input_data[i])
				self.backwardPropagation(input_labels[i], yi)

	def predict(self, input_data):
		output = []
		for i in range(len(input_data)):
			prediction = forwardPropagation(input_data[i])[0]
			result = np.where(prediction == np.max(prediction))
			output.append(result)
		return output

	def score(self, input_data, input_labels):
		result = predict(input_data)
		cnt = 0


		for i in range(len(input_data)):
			if input_labels[i] in result[i]:
				cnt += 1
		return cnt/len(input_data)

	def calculateError(self,input_labels, output):
		error = 0
		for i in range(len(input_labels)):
			y = input_labels[i]
			if y == 1:
				error += -log(output[i])
			else:
				error += -log(1-output[i])
		return error


import idx2numpy 
import numpy as np

# y_test = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')
# x_test = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')


y_train = idx2numpy.convert_from_file('train-labels-idx1-ubyte')
x_train = idx2numpy.convert_from_file('train-images-idx3-ubyte')


# # Load data from https://www.openml.org/d/554
# X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
# print ('ok')
# X = X / 255.

# # rescale the data, use the traditional train/test split
# X_train, X_test = X[:60000], X[60000:]
# y_train, y_test = y[:60000], y[60000:]

x_train = x_train / 255.
# x_test = x_test / 255.

X_train = []
# X_test = []

for i in range(len(x_train)):
	X_train.append(x_train[i].flatten())

# for i in range(len(x_test)):
# 	X_test.append(x_test[i].flatten())

X_train = np.array(X_train)
# X_test = np.array(X_test)
X_train = X_train[:2]
nn = Neural_Net(5,[784, 256, 128, 64, 10],'sigmoid',0.01)
nn.fit(input_labels = y_train, batch_size = 20, epochs = 10, input_data = X_train)
print (np.shape(X_train))







