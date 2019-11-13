from sklearn.neural_network import MLPClassifier
import idx2numpy 
import numpy as np
from neural_net import Neural_Net

y_test = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')
x_test = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')

y_train = idx2numpy.convert_from_file('train-labels-idx1-ubyte')
x_train = idx2numpy.convert_from_file('train-images-idx3-ubyte')

x_train = x_train / 255.
x_test = x_test / 255.

X_train = []
X_test = []

for i in range(len(x_train)):
	X_train.append(x_train[i].flatten())

for i in range(len(x_test)):
	X_test.append(x_test[i].flatten())

X_train = np.array(X_train)
y_train = y_train
X_test = np.array(X_test)

nn = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation = 'tanh', max_iter=100,
                    solver='adam', tol=1e-4, learning_rate_init=0.01)

nn.fit(X_train, y_train)

print("Test set score: %f" % nn.score(X_test, y_test))

train_X = idx2numpy.convert_from_file('train-images-idx3-ubyte')
train_Y = idx2numpy.convert_from_file('train-labels-idx1-ubyte')
test_X = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')
test_Y = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')

train_y = np.transpose(np.asmatrix(train_Y))
test_y = np.transpose(np.asmatrix(test_Y))

train_x = np.zeros((train_X.shape[0], train_X.shape[1]**2))
test_x = np.zeros((test_X.shape[0], test_X.shape[1]**2))

for i in range(train_X.shape[0]):
	train_x[i, :] = train_X[i].flatten()

for i in range(test_X.shape[0]):
	test_x[i, :] = test_X[i].flatten()

x = train_x[:300, :]
y = train_y[:300, :]

nn = Neural_Net(5,[784, 256, 128, 64, 10],'linear',0.01)
nn.fit(input_labels = y, batch_size = 10, epochs = 100, input_data = x)
print (nn.score(test_set, test_y))
