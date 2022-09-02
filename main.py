import math

import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm

use_gpu = torch.cuda.is_available()


# ========================================
#      loading datas



def scaleEachUnitaryDatas(datas,p=2):
  
    norms = datas.norm(dim=p, keepdim=True)
    return datas/norms


def QRreduction(datas):
    
    ndatas = torch.qr(datas.permute(0,2,1)).R
    ndatas = ndatas.permute(0,2,1)
    return ndatas

def Coles(ndatas, K, lam, lowrank=5):
    ndatas = QRreduction(ndatas)#to speed up svd,not necessary for the performance
    u, _, _ = torch.svd(ndatas)
    u = u[:, :, :lowrank]
    W = torch.abs(u.matmul(torch.transpose(u,dim0=2,dim1=1)))
    for i in range(W.shape[0]):
        W[i,:,:].squeeze().fill_diagonal_(0)
    isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(W, dim=-1,keepdim=True))
    W = W * isqrt_diag * torch.transpose(isqrt_diag,dim0=2,dim1=1)
    W1 = torch.ones_like(W)
    for i in range(W1.shape[0]):
        W1[i,:,:].squeeze().fill_diagonal_(0)
    isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(W1, dim=-1,keepdim=True))
    W1 = W1 * isqrt_diag * torch.transpose(isqrt_diag,dim0=2,dim1=1)
    lapReg1 = torch.transpose(ndatas,dim0=2,dim1=1).matmul(lam * W1 - W).matmul(ndatas)
    e, v1 = torch.linalg.eigh(lapReg1)
    return ndatas.matmul(v1[:, :, :K])


class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways
              
# ---------  GaussianModel
class GaussianModel(Model):
    def __init__(self, n_ways, lam):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None         # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam
        
    def clone(self):
        other = GaussianModel(self.n_ways)
        other.mus = self.mus.clone()
        return self

    def cuda(self):
        self.mus = self.mus.cuda()
        
    def initFromLabelledDatas(self, ndatas, n_runs, n_shot, n_queries, n_ways, n_nfeat):
        self.mus_ori = ndatas.reshape(n_runs, n_shot + n_queries, n_ways, n_nfeat)[:, :n_shot, ].mean(1)
        self.mus = self.mus_ori.clone()

    def updateFromEstimate(self, estimate, alpha):

        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-3):
        
        r = r.cuda()
        c = c.cuda()
        n_runs, n, m = M.shape
        P = torch.exp(- self.lam * M)
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)
                                         
        u = torch.zeros(n_runs, n).cuda()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P[:,:n_lsamples].fill_(0)
            P[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            P[:,:n_lsamples].fill_(0)
            P[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)
    
    def getProbas(self, ndatas, n_runs, n_ways, n_usamples, n_lsamples):
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = (ndatas.unsqueeze(2)-self.mus.unsqueeze(1)).norm(dim=3).pow(2)

        r = torch.ones(n_runs, n_lsamples + n_usamples)
        c = torch.ones(n_runs, n_ways) * (n_queries+n_shot)
       
        p_xj_test, _ = self.compute_optimal_transport(dist, r, c, epsilon=1e-3)
        p_xj = p_xj_test
        return p_xj

    def getProb(self, ndatas, n_runs, n_ways, n_usamples, n_lsamples):
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = (ndatas.unsqueeze(2) - self.mus.unsqueeze(1)).norm(dim=3).pow(2)
        p_xj = torch.zeros_like(dist)
        ind = torch.argmin(dist, dim=-1)
        p_xj.scatter_(2,ind.unsqueeze(2), 1)
        return p_xj

    def estimateFromMask(self, mask, ndatas, dis = False):
        if dis == True:
            mask = torch.zeros_like(mask).scatter_(2, mask.argmax(dim=-1).unsqueeze(-1), 1.)
        emus = mask.permute(0, 2, 1).matmul(ndatas)/(n_queries+n_shot)

        return emus

          
# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, alpha=None):
        
        self.verbose = False
        self.progressBar = False
        self.alpha = alpha
    
    def getAccuracy(self, probas):
        olabels = probas.argmax(dim=2)
        matches = labels.eq(olabels).float()
        acc_test = matches[:,n_lsamples:].mean(1)    

        m = acc_test.mean().item()
        pm = acc_test.std().item() *1.96 / math.sqrt(n_runs)
        return m, pm
    
    def performEpoch(self, model, ndatas, n_runs, n_ways, n_usamples, n_lsamples, epochInfo=None):
     
        p_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
        self.probas = p_xj
        
        if self.verbose:
            print("accuracy from filtered probas", self.getAccuracy(self.probas))
        
        m_estimates = model.estimateFromMask(self.probas,ndatas)
               
        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

        if self.verbose:
            op_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
            acc = self.getAccuracy(op_xj)
            print("output model accuracy", acc)
        
    def loop(self, model, ndatas, n_runs, n_ways, n_usamples, n_lsamples, n_epochs=20):
        
        self.probas = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
        if self.verbose:
            print("initialisation model accuracy", self.getAccuracy(self.probas))

        if self.progressBar:
            if type(self.progressBar) == bool:
                pb = tqdm(total = n_epochs)
            else:
                pb = self.progressBar
           
        for epoch in range(1, n_epochs+1):
            if self.verbose:
                # print("----- epoch[{:3d}]  lr_p: {:0.3f}  lr_m: {:0.3f}".format(epoch, self.alpha))
                print("----- epoch[{:3d}]".format(epoch))
            self.performEpoch(model, ndatas, n_runs, n_ways, n_usamples, n_lsamples, epochInfo=(epoch, n_epochs))
            if (self.progressBar): pb.update()

        op_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
        acc = self.getAccuracy(op_xj)
        return acc
    

if __name__ == '__main__':
# ---- data loading
    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs=10000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    
    import FSLTask
    cfg = {'shot':n_shot, 'ways':n_ways, 'queries':n_queries}
    FSLTask.loadDataSet("Res12_miniimagenet")
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    ndatas = ndatas.permute(0,2,1,3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1,1,n_ways).expand(n_runs,n_shot+n_queries,5).clone().view(n_runs, n_samples)
    
    # Power transform
    beta = 0.5
    #ndatas[:,] = torch.pow(ndatas[:,]+1e-6, beta) // need to uncomment this line for features based on WRN28
    ndatas = scaleEachUnitaryDatas(ndatas)
    ndatas = Coles(ndatas, 40, 100, lowrank=5)
    n_nfeat = ndatas.size(2)
    
    ndatas = scaleEachUnitaryDatas(ndatas)

    
    print("size of the datas...", ndatas.size())

    # switch to cuda
    ndatas = ndatas.cuda()
    labels = labels.cuda()
    
    #MAP
    lam = 10
    model = GaussianModel(n_ways, lam)
    model.initFromLabelledDatas(ndatas, n_runs, n_shot,n_queries,n_ways,n_nfeat)
    
    alpha = 0.2
    optim = MAP(alpha)

    optim.verbose=True
    optim.progressBar=True

    acc_test = optim.loop(model, ndatas, n_runs, n_ways, n_usamples, n_lsamples, n_epochs=40)
    
    print("final accuracy found {:0.3f} +- {:0.3f}".format(*(100*x for x in acc_test)))
    
    

