import argparse
import cPickle
import os
import time
import random
import h5py
from sklearn import preprocessing as pre
import torch
from torch.autograd import Variable
import eval
from eval import getScoreKnapsnck

from re_seg_model import Cholesky
import re_seg_model
# from re_seqdpp_clean import Cholesky
# import re_seqdpp_clean as re_seg_model

import numpy as np
import scipy.io as scio
import torch.optim as optim



parser = argparse.ArgumentParser(description='PyTorch summ model')

parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save', type=str,  default='./model/summeforshow',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################




def load_dataset_h5(data_dir, dataset,label_type):
    # loading data from a dataset in the format of hdf5 file
    feature = []
    label = []
    label2 = []
    file_name = data_dir + '/Data_' + dataset + '_google_p5.h5'
    f = h5py.File(file_name)
    vid_ord = np.sort(np.array(f['/ord']).astype('int32').flatten())
    for i in vid_ord:
        feature.append(np.matrix(f['/fea_' + i.__str__()]).astype('float32'))
        label.append(np.array(f['/gt_' + label_type.__str__() + '_' + i.__str__()]).astype('float32').flatten())
        label2.append(np.array(f['/gt_' + '1' + '_' + i.__str__()]).astype('float32').flatten())

    f.close()
    return feature, label

def subsets1(nums):
    res = []
    dfs(sorted(nums), 0, [], res)
    return res
    
def dfs(nums, index, path, res):
    res.append(path)
    for i in xrange(index, len(nums)):
        dfs(nums, i+1, path+[nums[i]], res)
def subsets(nums):
    res = [[]]
    for num in sorted(nums):
        res += [item+[num] for item in res]
    return res
model_type = 2
data_dir = './data/'    
[feature, label] = load_dataset_h5(data_dir, 'SumMe',model_type)

print feature[0].shape





    



videofealist = []
oracle_summarys = []



for i in range(len(feature)):
    oracle_summary = []
    for j in range(label[i].shape[0]):
        if label[i][j]==1:
            oracle_summary.append(j)
    oracle_summarys.append(oracle_summary)  
    videofea = np.asarray(np.transpose(feature[i]),dtype='float32')
  
    tmp = [videofea]    
    tmp = [Variable(torch.from_numpy(_), requires_grad=False) for _ in tmp]
    trvideofea = tmp[0]
    if args.cuda:
        trvideofea = trvideofea.cuda()
    videofealist.append(trvideofea)

  

iter = 20
re_times = 4

#testsummries = []
idx = range(25)
#2 
random.seed(args.seed)

# idx = [16, 20, 13, 22, 17, 9, 7, 11, 4, 0, 18, 2, 12, 21, 6, 19, 24, 15, 3, 10, 1, 23, 14, 8, 5]
# print idx
bestvalscore = [0 for _ in range(10)]
besttestscore = [0 for _ in range(10)]

# random.shuffle(idxvaltest)

trainid = 0
for testtime in range(10):

#     try:  
    random.shuffle(idx)
#         random.shuffle(idx)
    idxvaltest = idx[15:25]
    random.shuffle(idxvaltest)             
    seqmodel = re_seg_model.seqdpp()
    if args.cuda:
        seqmodel = seqmodel.cuda()
    else:
        seqmodel = seqmodel.cpu()
         
    optimizer = optim.Adam(seqmodel.parameters(), lr=args.lr)
    #optimizer = optim.SGD(seqmodel.parameters(), lr=args.lr,momentum=0.9)

    for i in range(iter):
 
        begin = time.time()
        lossall = 0
        for idindex in range(15):
             
            j = idx[idindex]
            if j==14:
                continue
            trainid = j
 
             
 
 
            loss = seqmodel(videofealist[j],oracle_summarys[j],args.cuda)
            shotscores = seqmodel.greedySearchFast(videofealist[j],args.cuda)
       
            f1score = getScoreKnapsnck(j,shotscores)
            loss = float(f1score)*loss
 
            optimizer.zero_grad()
            loss.backward()
            #lossiter.backward()        
            optimizer.step()
            print 'time : ',time.time()-begin
         
        totalvalscore = 0
        for validindex in range(5): 
            valid = idxvaltest[validindex]
            shotscores = seqmodel.greedySearchFast(videofealist[valid],args.cuda)
 
            #shotscores = np.random.random(len(shotscores)).tolist()
             
            f1score = getScoreKnapsnck(valid,shotscores)
            totalvalscore += f1score
        f1scoreval = totalvalscore/5   
        print 'valf1score: ',f1scoreval
        totaltestscore = 0
        for testindex in range(5,10): 
            testid = idxvaltest[testindex]
            shotscores = seqmodel.greedySearchFast(videofealist[testid],args.cuda)

            #shotscores = np.random.random(len(shotscores)).tolist()
             
            f1score = getScoreKnapsnck(testid,shotscores)
            totaltestscore += f1score
        f1scoretest = totaltestscore/5   
        print 'testf1score: ',f1scoretest
        if bestvalscore[testtime] <= f1scoreval:
            bestvalscore[testtime] = f1scoreval
            besttestscore[testtime] = f1scoretest
            with open(args.save+'.pt', 'wb') as f:
                torch.save(seqmodel, f)
#         testsummries.append(gensummtests)
    
    
    print 'bestval',bestvalscore
    print 'besttest',besttestscore
print 'bestval',bestvalscore
print 'besttest',besttestscore

besttestscore = np.array(besttestscore)
print 'mean',np.mean(besttestscore)
print 'std',np.std(besttestscore)



       
    
    
    
    
    
    

