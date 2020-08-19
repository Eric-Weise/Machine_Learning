# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from keras.datasets import mnist
from matplotlib import pyplot
import matplotlib.pyplot as plt

size = 28
numLabels = 10
pixels = size*numLabels

train = np.loadtxt('mnist_train_6000.csv', delimiter= ',')
test = np.loadtxt('mnist_test_1000.csv', delimiter=',')

no_of_different_labels = 10
test[test==255]
test.shape
fac = 0.99 / 255
train_imgs = np.asfarray(train[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test[:, 1:]) * fac + 0.01
train_labels = np.asfarray(train[:, :1])
test_labels = np.asfarray(test[:, :1])
lr = np.arange(10)

for label in range(10):
    one_hot = (lr==label).astype(np.int)
    print("label: ", label, " in one-hot representation: ", one_hot)

lr = np.arange(no_of_different_labels)

# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99

for i in range(10):
    img = train_imgs[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()