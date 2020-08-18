#https://github.com/jdwittenauer/ipython-notebooks/blob/master/notebooks/ml/ML-Exercise3.ipynb
#Paul Jean-Baptiste & Eric Weise
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    
    return np.sum(first - second) / (len(X)) + reg
    

def gradient_with_loop(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])
    
    return grad


def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y
    
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    
    # intercept gradient is not regularized
    #grad[0] = np.sum(np.multiply(error, X[:,0])) / len(X)
    grad[0] = grad[0] - learningRate / len(X) * theta[0]
    
    return np.array(grad).ravel()


def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]
    
    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))
    
    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    
    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))
        
        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i-1,:] = fmin.x
    
    return all_theta


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]
    
    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    
    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    
    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)
    
    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)
    
    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1
    
    return h_argmax

def predict_oneinstance(X, all_theta):
	rows = X.shape[0]
	params = X.shape[1]
	errorPrediction = []
	print("\nPrediction of individual models: \n",X,"\n")
	#calculate prediction
	for i in (range(rows)):
		errorPrediction.append([])
		for z in (range(1)):
			for j in (range(params)):
				errorPrediction[i].append(1-(1)/(1+np.exp(all_theta[0][j]+all_theta[1][j]*(X[i][z])+all_theta[2][j]*(X[i][z]))))
    # same as before, insert ones to match the shape
	X = np.insert(errorPrediction, 0, values=np.ones(rows), axis=1)
    # convert to matrices
	X = np.matrix(errorPrediction)
	print("Normalized Prediction of individual models:\n",X,"\n")
	print("Final Predicted Model Class\n",np.max(X))
	return X

def plotGraph(X, all_theta):
	dframe = pd.read_csv("Table7_11.txt", sep=",", names=["SPEND", "FREQ", "TYPE"])
	y = np.array(dframe["TYPE"])
	plotX = []
	for i in range(len(X)):
		plotX.append(X[i][0])
	plotX2 = []
	for i in range(len(X)):
		plotX2.append(X[i][1])
	    
	plotFrame = pd.DataFrame(columns = ['ONE', 'TWO', 'TARGET'])
	plotFrame['ONE'] = plotX
	plotFrame['TWO'] = plotX2
	plotFrame['TARGET'] = dframe['TYPE']
    
	one = plotFrame[plotFrame['TARGET'].isin([1])]
	two = plotFrame[plotFrame['TARGET'].isin([2])]
	three = plotFrame[plotFrame['TARGET'].isin([3])]
    
	fig, ax = plt.subplots(figsize=(12,8))
	ax.scatter(one['ONE'], one['TWO'], s=50, c='b', marker='o', label='f=1.32+1.74s')    
	ax.scatter(two['ONE'], two['TWO'], s=50, c='r', marker='+', label='f=1.23-2.43s')
	ax.scatter(three['ONE'], three['TWO'], s=50, c='k', marker='x', label='f=0.32-0.04s')
	ax.legend()
	ax.set_xlabel('SPEND')
	ax.set_ylabel('FREQ')
	
	plt.scatter(X, X, y, alpha = .0001)
	for theta in all_theta:
		j = np.array([X.min(), X.max()])
		k = -(j*theta[1] + theta[0]) / theta[2]
		plt.plot(j, k/3, color='k', linestyle = "-")
	return 0

def main() :
	
	dframe = pd.read_csv("Table7_11.txt", sep=",", names=["SPEND", "FREQ", "TYPE"])
	X = np.array(dframe[["SPEND", "FREQ"]])
	y = np.array(dframe["TYPE"])
    
	X1 = X
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler1 = MinMaxScaler(feature_range=(0, 1))
    
	X = scaler.fit_transform(X)
	X1 = scaler1.fit_transform(X1)

	rows = X.shape[0]
	parameters = X.shape[1]
    
	numLabels = 3
	all_theta = np.zeros((numLabels, parameters +1))
	all_theta = one_vs_all(X, y, numLabels, 0.0001)
	print("Y: \n", y)
	print("Final weights: \n", all_theta)
#
#	# predict all original dataset using the trained models
	y_pred = predict_all(X, all_theta)
	correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
	accuracy = (sum(map(int, correct)) / float(len(correct)))
	print('accuracy = {0}%'.format(accuracy * 100))

	predict_oneinstance(X1,all_theta)
	plotGraph(X,all_theta)
#invoke the main
main()
