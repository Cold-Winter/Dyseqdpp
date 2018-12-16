from datashape.typesets import scalar
import torch
from torch.autograd import Variable


import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as trnn


class Cholesky(torch.autograd.Function):  # @UndefinedVariable
    @staticmethod
    def forward(ctx, a):
        l = torch.potrf(a, False)
        ctx.save_for_backward(l)
        return l
    @staticmethod
    def backward(ctx, grad_output):
        l, = ctx.saved_variables

        linv =  l.inverse()
        
        inner = torch.tril(torch.mm(l.t(),grad_output))*torch.tril(1.0-Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        
        return s
def subsets(nums):
    res = [[]]
    for num in sorted(nums):
        res += [item+[num] for item in res]
    return res

class seqdpp(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self):
        super(seqdpp, self).__init__()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.ufea = nn.Linear(1024,256)
        self.wforu = nn.Linear(256,256)
        self.relulinear = nn.Linear(256,256)
            
        self.softmax = nn.Softmax()
        self.wl = nn.Linear(1024,256)
        self.predfc = nn.Linear(256,10)
        self.seglen = range(5,15)
        
        #self.ufea.weight.data = scalar*self.ufea.weight.data.normal_(0,1)
        #self.wforu.weight.data.normal_(0.001,0.02)
    def det(self,omig):
        temp = Cholesky.apply(omig).diag().prod()
        return temp*temp

    def forward(self,videoseglist,oracle_summmary,cudaflag):
        sequencelen = videoseglist.size()[0]   
        self.lmat = torch.mm(self.relulinear(self.relu(self.wforu(self.relu(self.ufea(videoseglist))))),torch.transpose(self.relulinear(self.relu(self.wforu(self.relu(self.ufea(videoseglist))))),0,1))       
        lastseg = 0
        accumseg = 0
        prednum = 10
        invideo = True
        probsum = 0
        while(invideo):
            if accumseg + prednum>=sequencelen:
                prednum = sequencelen-accumseg
                invideo = False
            curr_sum = []
            curr_video = []
            predfeaid = []
            endseg = accumseg+prednum
            for sumid in oracle_summmary:
                if sumid < endseg:
                    predfeaid.append(sumid)
                if sumid >= accumseg-lastseg and sumid < endseg:
                    curr_sum.append(sumid)
                    curr_video.append(sumid)
                      
            for segvid in range(accumseg,endseg):
                if segvid not in curr_sum:
                    curr_video.append(segvid)
                    predfeaid.append(segvid)
                  
            curr_video = sorted(curr_video)
            predfeaid = torch.LongTensor(sorted(predfeaid))
            if cudaflag:
                predfeaid = predfeaid.cuda()
            predfea = videoseglist[predfeaid]
            poolfea = torch.max(predfea,0)[0].view(1,-1)
            pred = self.softmax(self.predfc(self.relu(self.wl(poolfea))))
            lastseg = prednum    
            prednum = self.seglen[torch.max(pred,1)[1].data[0]]
            
            #print omigup            
            dignose = np.zeros((len(curr_video),len(curr_video)))
            oneindex = []
            for dindex,dsegvid in enumerate(curr_video):
                if dsegvid >= accumseg:
                    oneindex.append(dindex)
            dignose[oneindex,oneindex]=1
            
            accumseg += lastseg


            dignose = Variable(torch.from_numpy(np.asarray(dignose,dtype='float32')),requires_grad=True)
            if cudaflag:
                dignose = dignose.cuda()                                
            omigdown = self.lmat[curr_video,][:,curr_video] + dignose
            weight = 0
            if len(curr_sum) == 0:
                detup = 1
            else:
                omigup = self.lmat[curr_sum,][:,curr_sum]
                detup = self.det(omigup)
            detdown = self.det(omigdown)           
            probseg = detup/detdown                    
            #break
#             if len(curr_sum)<=5:
#                 probsum -= torch.log(probseg)+torch.log(torch.max(pred,1)[0])
            probsum -= torch.log(probseg)+torch.log(torch.max(pred,1)[0])          
            #probsum -= weight * torch.log(probseg)
        return probsum
    
    def sampleFromSeqdpp(self,videoseglist,cudaflag,samplenum):
        lmat = torch.mm(self.relulinear(self.relu(self.wforu(self.relu(self.ufea(videoseglist))))),torch.transpose(self.relulinear(self.relu(self.wforu(self.relu(self.ufea(videoseglist))))),0,1))    
        if cudaflag:
            lmat = lmat.cpu().data.numpy()
        else:
            lmat = lmat.data.numpy()
        sequencelen = videoseglist.size()[0]
        samplesumms=[]
        for sampletime in range(samplenum):  
                   
            lastseg = 0
            accumseg = 0
            prednum = 10
            invideo = True
            lastvideosumm = []
            allvideosumm = [] 
            while(invideo):
                if accumseg + prednum>=sequencelen:
                    prednum = sequencelen-accumseg
                    invideo = False
                curr_sum = []
                curr_video = []
                predfeaid = []
                endseg = accumseg+prednum
                summlist = subsets(range(accumseg,endseg))
    
                prolist = []
                for summ in summlist:
                    curr_sum = []
                    for sumid in summ:
                        curr_sum.append(sumid)
                    for item in lastvideosumm:
                        curr_sum.append(item)  
                    if len(curr_sum) == 0:
                        detup = 1
                    else:
                        omigup = lmat[curr_sum,][:,curr_sum]
                        detup =  np.linalg.det(omigup)
                        
                    prolist.append(detup)
                    
                prosum = sum(prolist)        
                prolist = [_ / prosum for _ in prolist]
                sampleid = np.random.choice(len(prolist),1,prolist)[0]
#                 lenflag = True
#                 while(lenflag):   
#                     if prosum == 0:
#                         sampleid = np.random.choice(len(prolist),1)[0]
#                     else:
#                         prolist = [_ / prosum for _ in prolist]
#                         sampleid = np.random.choice(len(prolist),1,prolist)[0]
#                     if len(summlist[sampleid])<=5:
#                         lenflag = False
                lastvideosumm = []
                if  len(summlist[sampleid]) != 0:
                    for item in summlist[sampleid]:
                        lastvideosumm.append(item)
                        allvideosumm.append(item)
                          
                for segvid in range(accumseg,endseg):
                    if segvid not in summlist[sampleid]:
                        predfeaid.append(segvid)
                predfeaid = allvideosumm+predfeaid
                predfeaid = torch.LongTensor(sorted(predfeaid))
                if cudaflag:
                    predfeaid = predfeaid.cuda()
                predfea = videoseglist[predfeaid]
                poolfea = torch.max(predfea,0)[0].view(1,-1)
                pred = self.softmax(self.predfc(self.relu(self.wl(poolfea))))
                lastseg = prednum    
                prednum = self.seglen[torch.max(pred,1)[1].data[0]]
                                  
                accumseg += lastseg

            samplesumms.append(allvideosumm)

        return samplesumms
    def greedySearchFast(self,videoseglist,cudaflag):
        lmat = torch.mm(self.relulinear(self.relu(self.wforu(self.relu(self.ufea(videoseglist))))),torch.transpose(self.relulinear(self.relu(self.wforu(self.relu(self.ufea(videoseglist))))),0,1))    
        if cudaflag:
            lmat = lmat.cpu().data.numpy()
        else:
            lmat = lmat.data.numpy()
        sequencelen = videoseglist.size()[0]
        shotscores = []
       
        lastseg = 0
        accumseg = 0
        prednum = 10
        invideo = True
        lastvideosumm = []
        allvideosumm = [] 
        while(invideo):
            if accumseg + prednum>=sequencelen:
                prednum = sequencelen-accumseg
                invideo = False
            curr_sum = []
            curr_video = []
            predfeaid = []
            endseg = accumseg+prednum

            summlist = subsets(range(accumseg,endseg))

            prolist = []
            for summ in summlist:
                curr_sum = []
                for sumid in summ:
                    curr_sum.append(sumid)
                for item in lastvideosumm:
                    curr_sum.append(item)  
                if len(curr_sum) == 0:
                    detup = 1
                else:
                    omigup = lmat[curr_sum,][:,curr_sum]
                    detup =  np.linalg.det(omigup)               
                prolist.append(detup)
                
            for shotid in range(accumseg,endseg):
                shotidlist = [shotid]
                omigshot = lmat[shotidlist,][:,shotidlist]
                detshot =  np.linalg.det(omigshot)
                shotscores.append(detshot/sum(prolist))
                
                
                
                
            
            lastvideosumm = []
            maxpro = 0
            maxsummid = 0
            for summindex in range(len(summlist)):
                if prolist[summindex] >= maxpro:
                    maxsummid = summindex
                    maxpro = prolist[summindex]
            if  len(summlist[maxsummid]) != 0:
                for item in summlist[maxsummid]:
                    lastvideosumm.append(item)
                    allvideosumm.append(item)
                      
            for segvid in range(accumseg,endseg):
                if segvid not in summlist[maxsummid]:
                    predfeaid.append(segvid)
            predfeaid = allvideosumm+predfeaid
            predfeaid = torch.LongTensor(sorted(predfeaid))
            if cudaflag:
                predfeaid = predfeaid.cuda()
            predfea = videoseglist[predfeaid]
            poolfea = torch.max(predfea,0)[0].view(1,-1)
            pred = self.softmax(self.predfc(self.relu(self.wl(poolfea))))
            lastseg = prednum    
            prednum = self.seglen[torch.max(pred,1)[1].data[0]]

                              
            accumseg += lastseg



        return shotscores
    
            
            
                    
                    
        
                    
             
        #for i in range(video

