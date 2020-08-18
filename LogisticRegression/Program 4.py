# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pylab import plot, show, xlabel, ylabel

dataframe = pd.read_csv('Table7_7.txt', sep=",",header = None, names=["ID", "RPM", "Vibration", "Status"])

errorframe = pd.DataFrame(columns = ["Target", "Prediction", "Error", "ErrorSquared"])
errorDelta = pd.DataFrame(columns = ["Target","W[0]","W[1]","W[2]"])

#Normalize
def rangenorm(scores):
    lower = -1
    upper = 1
    normlist = []
    for i in range(len(scores)):
        norm = ((scores[i] - min(scores)) / (max(scores) - min(scores))) * (upper - lower) + lower
        #stand = ((norm - min(scores)) / max(scores)) / stdev(scores)
        normlist.append(norm)
    return normlist

def sumErrorDelta(delta):
    sum =0
    for x in range(5):
        sum += delta[x]
    return sum

#Normalize values in original dataset
rpm = rangenorm(np.array(dataframe["RPM"]))
dataframe["RPM"] = rpm
vib = rangenorm(np.array(dataframe["Vibration"]))
dataframe["Vibration"] = vib

weights = [-2.9465, -1.0147, 2.161]
alpha = 0.02

errorframe["Target"] = dataframe["Status"]
errorDelta["Target"] = dataframe["Status"]

print("\nOriginal dataset: \n", dataframe[:5],"\n")
print("\nInitial Weights:", weights,"\n")

sumDeltas = [0,0,0]

iterations =2000
#for each iteration:
for j in range(iterations):
    #for each row, calculate values:
    for i in range(len(dataframe["ID"])):
        errorframe["Prediction"][i] = 1-(1)/(1+np.exp(weights[0]+weights[1]*(dataframe["RPM"][i])+weights[2]*(dataframe["Vibration"][i])))
        errorframe["Error"][i] = (1-errorframe["Prediction"][i])
        errorframe["ErrorSquared"][i] = (errorframe["Error"][i])*(errorframe["Error"][i])
        
        errorDelta["W[0]"][i] = errorframe["Error"][i]*errorframe["Prediction"][i]*(1-errorframe["Prediction"][i])*errorframe["Target"][i]
        errorDelta["W[1]"][i] = errorframe["Error"][i]*errorframe["Prediction"][i]*(1-errorframe["Prediction"][i])*dataframe["RPM"][i]
        errorDelta["W[2]"][i] = errorframe["Error"][i]*errorframe["Prediction"][i]*(1-errorframe["Prediction"][i])*dataframe["Vibration"][i]
    #calculate new weights:
    sumDeltas[0] = sumErrorDelta(errorDelta["W[0]"])
    sumDeltas[1] = sumErrorDelta(errorDelta["W[1]"])
    sumDeltas[2] = sumErrorDelta(errorDelta["W[2]"])
    weights[0] = weights[0]+alpha*sumDeltas[0]
    weights[1] = weights[1]+alpha*sumDeltas[1]
    weights[2] = weights[2]+alpha*sumDeltas[2]
    if(j>0 and j<5):
        print("New weights after iteration ",j,": ", weights, "\n")
    #Print intermediate results for first 5 training instances
    if(j<5):
        print("Errors [The first five] : \n", errorframe[:5],"\n")
        print("ErrorDelta [The first five] : \n", errorDelta[:5],"\n")
    #Plot logistic models for these iterations:
    #Print final sum of squared errors after 2000 iterations
    if(j==1999):
            print("Final weights after iteration ",j+1,": ", weights, "\n")
    if (j==0 or j==9 or j==200 or j==499 or j==1999):
        x=1
#        plot(arange(iterations), sumDeltas)
#        xlabel('RPM')
#        ylabel('Vibration')
#        show()