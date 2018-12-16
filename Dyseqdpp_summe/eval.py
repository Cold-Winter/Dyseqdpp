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
namevidlist = open('namevid.txt').readline().strip().split('\t')
namevidlist = [name[1:-1] for name in namevidlist]


def getOneScore(vname,gensummary):  
    dir = './evalSumMe/GT/'
    filename = dir+vname+'.mat'
    userscore = scio.loadmat(filename)['user_score']
    nframe,nusers = userscore.shape
    genselect = np.zeros(nframe)
    for i in gensummary:
        genselect[i*15:(i+1)*15] = 1
    user_intersection = np.zeros(nusers)
    for i in range(nusers):
        onescore = userscore[:,i]
        user_intersection[i] = np.count_nonzero(genselect*onescore)

    recall = user_intersection / np.count_nonzero(userscore, 0)
    precision = user_intersection / np.count_nonzero(genselect)
    fmeasure = 2*recall*precision/(recall+precision)
    fmeasure[np.isnan(fmeasure)] = 0
    f1score = np.max(fmeasure)
    return f1score
def getmaxscore(testidlist,gensummaries): 
    f1score = 0      
    for i,v in enumerate(testidlist):
        vname = namevidlist[v] 
        f1score += getOneScore(vname,gensummaries[i])
#         print getOneScore(vname,[20,30,40,50,55,60])
#         f1score += getOneScore(vname,[20,30,40,50,55,60])
    return f1score/len(testidlist)




 

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
def getScoreKnapsnck(vid,shotscore,rate = 0.15):
    vname = namevidlist[vid] 
#     print vname
    dir = './evalSumMe/GT/'
    filename = dir+vname+'.mat'
    userscore = scio.loadmat(filename)['user_score']
    nframe,nusers = userscore.shape
    framescore = np.zeros(nframe)
    for i,value in enumerate(shotscore):
        if (i+1)*15>=nframe:
            break
        framescore[i*15:(i+1)*15] = value

    
    segboundary = scio.loadmat('./data/shot_SumMe.mat')['shot_boundaries'][:,0][vid][0]
#     print segboundary

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
    #gensumm
#     folder = './SumMe/frames/'
#     gensummfolder = 'seqdpp/'+vname
#     print gensummfolder
#     if not os.path.exists(gensummfolder):
#         os.makedirs(gensummfolder)
#     for seg in reconstruction:
#         begin = segboundary[seg]
#         end = segboundary[seg+1]
#         for frameid in range(begin,end):
#             
#             framefile = folder+vname+'.mp4'+"/frame%07d" % (frameid)+'.jpg'
#             os.system('cp \"'+framefile+'\" \"'+gensummfolder+'\"')
        
  
    genselect = np.zeros(nframe)
    for i in reconstruction:
        genselect[segboundary[i]:segboundary[i+1]] = 1 
    
    
#     print sum(genselect)/nframe

    
    user_intersection = np.zeros(nusers)
    for i in range(nusers):
        onescore = userscore[:,i]
        user_intersection[i] = np.count_nonzero(genselect*onescore)

    recall = user_intersection / np.count_nonzero(userscore, 0)
    precision = user_intersection / np.count_nonzero(genselect)
    fmeasure = 2*recall*precision/(recall+precision)
    fmeasure[np.isnan(fmeasure)] = 0
    f1score = np.max(fmeasure)
    return f1score
    


    

    
    
    
        
        
# f1result = getmaxscore(range(25),[range(25) for i in range(25)])
# print f1result
# totalscore = 0
# for i in range(25):
#     shotscore = np.random.random(301).tolist()
#     totalscore += getScoreKnapsnck(i,shotscore)
# print totalscore/25


if __name__=='__main__':  
    number=5  
    capacity=10  
    weight=[2,2,6,5,4]  
    value=[0.6,0.3,0.5,0.4,0.6]  
    res=bag(number,capacity,weight,value)  
    show(number,capacity,weight,res)  
    


            
            
            
            
            
        
        
            
        
    

        

       
    
    
    
    
    
    

