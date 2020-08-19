# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from math import sqrt

dataframe = pd.read_csv("CPI.txt", header = None)
dataframe.columns = ["CountryID", "LifeExp", "Top10Income", "InfantMort", "MilSpend", "SchoolYears", "CPI"]
dframe = pd.DataFrame(dataframe, columns=['LifeExp','Top10Income','InfantMort','MilSpend', 'SchoolYears'])
row = np.array(dframe[["LifeExp", "Top10Income", "InfantMort", "MilSpend", "SchoolYears"]])
query = [67.62, 31.68, 10.00, 3.87, 12.90]
k=3

def euclidean_distance(row, query):
    distList = []
    for i in range(len(dframe)):
        distance = 0.0
        for j in range(len(row[5])):
            distance += (query[j] - row[i][j])**2
        distList.append(sqrt(distance))
    return distList
distance = euclidean_distance(row, query)
distFrame = pd.DataFrame(dataframe, columns=['CountryID', 'Euclid', 'CPI', 'Weight', 'W*CPI'])

distFrame['Euclid'] = distance
cpi = np.array(distFrame['CPI'])

index = np.argpartition(distance, range(k))

def NearestNeighbor(distance, k, cpi, index):
    CPI = 0.0
    kSmallest = 0.0
    for i in range(k):
        kSmallest += cpi[index[:k]][i]
    CPI = kSmallest / k
    return CPI
NNCPI3 = NearestNeighbor(distance, k, cpi, index)

print("\n CPI for ",k,"-NN: ", round(NNCPI3,4), sep="")

k=16

NNCPI16 = NearestNeighbor(distance, k, cpi, index)

def WeightedReciprocal(distFrame, cpi):
    weight=[]
    recip=[]
    for i in range(len(distFrame["Euclid"])):
        weight.append(1/(distFrame["Euclid"][i])**2)
        recip.append(weight[i]*cpi[i])
    distFrame['Weight'] = weight  
    distFrame['W*CPI'] =  recip
    return distFrame

def sorted(distFrame):
    distFrame_reindex = distFrame.reindex(index = list(index))
    print("\n", distFrame_reindex)

WeightedReciprocal(distFrame,cpi)
sorted(distFrame.round(4))


print("\n CPI for ",k,"-NN: ", round(NNCPI16,4), sep="")