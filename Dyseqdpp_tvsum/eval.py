#coding=utf-8 
import argparse
import cPickle
import math
import os
import time

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn import preprocessing as pre
import torch
from torch.autograd import Variable

from re_seg_model import Cholesky
import h5py

import re_seg_model
import numpy as np
import scipy.io as scio
import torch.nn as nn
import torch.optim as optim
from dtw import dtw

np.seterr(divide='ignore', invalid='ignore')




###############################################################################
# Load data
###############################################################################


 

# f1result = getmaxscore(range(25),[])
# print f1result


def knapsack(items, maxweight):

    bestvalues = [[0] * (maxweight + 1)
                  for i in xrange(len(items) + 1)]

    for i, (value, weight) in enumerate(items):

        i += 1
        for capacity in xrange(maxweight + 1):

            if weight > capacity:
                bestvalues[i][capacity] = bestvalues[i - 1][capacity]
            else:

                candidate1 = bestvalues[i - 1][capacity]
                candidate2 = bestvalues[i - 1][capacity - weight] + value

                bestvalues[i][capacity] = max(candidate1, candidate2)

    reconstruction = []
    i = len(items)
    j = maxweight
    while i > 0:
        if bestvalues[i][j] != bestvalues[i - 1][j]:
            reconstruction.append(i-1)
            j -= items[i - 1][1]
        i -= 1

    reconstruction.reverse()

    return bestvalues[len(items)][maxweight], reconstruction

def getselect(framescore,segboundary,nframe,rate=0.18):
    shotnumber = segboundary.shape[0]-1
    segscore = np.zeros(shotnumber)
    weightlist = []
    for i in range(shotnumber):
        segscore[i]= np.mean(framescore[segboundary[i]:segboundary[i+1]])
        weightlist.append(segboundary[i+1]-segboundary[i])
    valuelist = segscore.tolist()
    
    capacity = int(nframe*rate)

    items = [[valuelist[i],weightlist[i]] for i in range(len(valuelist))]
    
    bestvalue, reconstruction = knapsack(items, capacity) 
  
    genselect = np.zeros(nframe)
    for i in reconstruction:
        genselect[segboundary[i]:segboundary[i+1]] = 1
        
    return genselect

filename ='./evalTVSum/tvsum50score.mat'
userscore_global = scio.loadmat(filename)['tvsum50']['user_anno'][0]
segboundarylist_global = scio.loadmat('./data/shot_TVSum.mat')['shot_boundaries'][:,0]
userselectlist = []
for videoid in range(50):
    segboundary_global = segboundarylist_global[videoid][0]
    userselect = np.zeros((userscore_global[videoid].shape[0],userscore_global[videoid].shape[1]))
    for i in range(userscore_global[videoid].shape[1]):
        userselect[:,i] = getselect(userscore_global[videoid][:,i],segboundary_global,userscore_global[videoid].shape[0],0.18)
    userselectlist.append(userselect)
    
def getScoreKnapsnck(vid,shotscore): 
    filename ='./evalTVSum/tvsum50score.mat'
    
    userscore = scio.loadmat(filename)['tvsum50']['user_anno'][0][vid]


    nframe,nusers = userscore.shape
    framescore = np.zeros(nframe)
    for i,value in enumerate(shotscore):
        if (i+1)*15>=nframe:
            break
        framescore[i*15:(i+1)*15] = value


    segboundary = scio.loadmat('./data/shot_TVSum.mat')['shot_boundaries'][:,0][vid][0]

    
    genselect = getselect(framescore,segboundary,nframe,0.15)
    
#     userselect = np.zeros((userscore.shape[0],userscore.shape[1]))
#     for i in range(userscore.shape[1]):
#         userselect[:,i] = getselect(userscore[:,i],segboundary,nframe,0.18)
    userselect = userselectlist[vid]
    
#     print sum(genselect)/nframe
#     shotnumber = segboundary.shape[0]-1
#     shotselect=[False for i in range(shotnumber)]

#     shotselect[0]= True
#     shotselect[1]= True
#     genselect = np.zeros(nframe)
#     for i,v in enumerate(shotselect):
#         if v:
#             genselect[segboundary[i]:segboundary[i+1]] = 1


    user_intersection = np.zeros(nusers)
    for i in range(nusers):
        onescore = userselect[:,i]
        user_intersection[i] = np.count_nonzero(genselect*onescore)

    recall = user_intersection / np.count_nonzero(userselect, 0)
    precision = user_intersection / np.count_nonzero(genselect)
    fmeasure = 2*recall*precision/(recall+precision)
    fmeasure[np.isnan(fmeasure)] = 0
    f1score = np.mean(fmeasure)
    return f1score
    


    

    
    
    
        
        
# f1result = getmaxscore(range(25),[range(25) for i in range(25)])
# print f1result
# totalscore = 0
# for i in range(50):
#     shotscore = np.random.random(600).tolist()
#     totalscore += getScoreKnapsnck(i,shotscore)
# print totalscore/50
# print getScoreKnapsnck(0,np.random.random(301).tolist())

# if __name__=='__main__':  
#     number=5  
#     capacity=10  
#     weight=[2,2,6,5,4]  
#     value=[0.6,0.3,0.5,0.4,0.6]  
#     res=bag(number,capacity,weight,value)  
#     show(number,capacity,weight,res) 

    


            
            
            
            
            
        
        
            
        
    

        

       
    
    
    
    
    
    

