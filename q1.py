import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

import idx2numpy 
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_test = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')
x_test = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')


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
x_test = x_test / 255.

X_train = []
X_test = []

for i in range(len(x_train)):
	X_train.append(x_train[i].flatten())

for i in range(len(x_test)):
	X_test.append(x_test[i].flatten())

X_train = np.array(X_train)
X_test = np.array(X_test)
print (np.shape(X_train))

# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(50,), activation = 'logistic', max_iter=10,
                    solver='adam', tol=1e-4, learning_rate_init=.01)

mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

# fig, axes = plt.subplots(4, 4)
# # use global min / max to ensure all weights are shown on the same scale
# vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
# for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
#     ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
#                vmax=.5 * vmax)
#     ax.set_xticks(())
#     ax.set_yticks(())

# plt.show()