
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
import numpy as np
import pickle
import torchvision
import numpy as np
import time
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import os
import sys
import argparse
import torch.nn as nn
import torch.nn.functional as F
import pickle

class APGDAttack_targeted():
    def __init__(self, model, n_iter=100, norm='Linf', n_restarts=1, eps=None,
                 seed=0, eot_iter=1, rho=.75, verbose=True, device='cuda',
                 n_target_classes=9,n_iter_2_p=.22, n_iter_min_p=.06, size_decr_p=.03,loss='ce'):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.target_class = None
        self.device = device
        self.n_target_classes = n_target_classes
        self.n_iter_2, self.n_iter_min, self.size_decr = max(int(n_iter_2_p * self.n_iter), 1), max(int(n_iter_min_p * self.n_iter), 1), max(int(size_decr_p * self.n_iter), 1)
        self.loss=loss

    def check_oscillation(self, x, j, k, y5, k3=0.5):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
          t += x[j - counter5] < x[j - counter5 - 1]
        return t <= k*k3*np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def ce_loss_targeted(self, x, y, y_target):
        celoss=nn.CrossEntropyLoss(reduce=False, reduction='none')
        return celoss(x, y_target)
    def md_loss_targeted(self, x, y, y_target):
        x_target=torch.unsqueeze(x[np.arange(x.shape[0]), y_target],1).expand(-1,x.shape[1])
        return torch.sum(nn.functional.relu(x+1e-15-x_target),dim=1)
    def dlr_loss_targeted(self, x, y, y_target):
        x_sorted, ind_sorted = x.sort(dim=1)  
        return -(x[np.arange(x.shape[0]), y] - x[np.arange(x.shape[0]), y_target]) / (x_sorted[:, -1] - .5 * x_sorted[:, -3] - .5 * x_sorted[:, -4] + 1e-12)
    def cw_loss_targeted(self, x, y, y_target):
        x_target=x[np.arange(x.shape[0]), y_target]
        target_holes=torch.zeros(x.shape,device=torch.device('cuda'))
        target_holes[np.arange(x.shape[0]), y_target]=1e8
        x_max_not_y,_=torch.max(x-target_holes,dim=1)
        return nn.functional.relu(x_max_not_y-x_target+50)

    def attack_single_run(self, x_in, y_in,target_offset=None,batch_idx=None,measure=False,measure_results_dir=''):
        print  ('Auto-PGD '+self.loss+' loss')
        if self.loss=='ce':
            self.loss_targeted=self.ce_loss_targeted
        if self.loss=='cw':
            self.loss_targeted=self.cw_loss_targeted
        if self.loss=='dlr':
            self.loss_targeted=self.dlr_loss_targeted
        if self.loss=='md':
            self.loss_targeted=self.md_loss_targeted
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        unit=torch.arange(x.size()[0]).to(self.device)
        if self.verbose:
            print('parameters: ', self.n_iter, self.n_iter_2, self.n_iter_min, self.size_decr)
        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        x_best_adv=torch.round(x_best_adv*255)/255.
        loss_steps = torch.zeros([self.n_iter, x.shape[0]])
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)
        output = self.model(x)
        if (target_offset is None):
            y_target = output.sort(dim=1)[1][:, -self.target_class]
        else:
            y_target=(y+target_offset) % 10
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                loss_indiv = self.loss_targeted(logits, y, y_target)
                loss = loss_indiv.sum()
            grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
        grad /= float(self.eot_iter)
        grad_best = grad.clone()
        acc=logits.detach().max(1)[1] != y_target
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()
        step_size = self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0
        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0
        losses=[]
        step_sizes=[]
        predicts=[]
        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                a = 0.75 if i > 0 else 1.0
                if self.norm == 'Linf':
                    x_adv_1 = x_adv - step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a), x - self.eps), x + self.eps), 0.0, 1.0)
                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)
                x_adv = x_adv_1 + 0.
            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                    loss_indiv = self.loss_targeted(logits, y, y_target)
                    loss = loss_indiv.sum()
                grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            grad /= float(self.eot_iter)
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, loss_best.sum()))
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1.cpu() + 0
              ind = (y1 < loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0
              counter3 += 1
              if counter3 == k:
                  fl_oscillation = self.check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=self.thr_decr)
                  fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() <= loss_best.cpu().numpy())
                  fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                  reduced_last_check = np.copy(fl_oscillation)
                  loss_best_last_check = loss_best.clone()
                  if np.sum(fl_oscillation) > 0:
                      step_size[u[fl_oscillation]] /= 2.0
                      n_reduced = fl_oscillation.astype(float).sum()
                      fl_oscillation = np.where(fl_oscillation)
                      x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                      grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                  counter3 = 0
                  k = np.maximum(k - self.size_decr, self.n_iter_min)
            pred=self.model(torch.round(x_adv*255)/255.).max(1)[1] != y_target
            unchangable=torch.masked_select(unit,torch.logical_not(pred))
            x_best_adv[unchangable]=x_adv[unchangable]
            x_best_adv=torch.round(x_best_adv*255)/255.
            x_best_adv=torch.clamp(x_best_adv,0,1)
            print ((self.model(x_best_adv).max(1)[1] != y_target).sum().item())        
        x_best_adv = torch.clamp(torch.min(torch.max(x_best_adv, x - self.eps), x + self.eps), 0.0, 1.0)
        x_best_adv=torch.round(x_best_adv*255)/255.
        acc= ((self.model(x_best_adv).max(1)[1] != y_target))
        return x_best_adv, acc, loss_best, x_best_adv



class CGD():
    def __init__(self, model, n_iter=100, norm='Linf', n_restarts=1, eps=None,
                 seed=0, eot_iter=1, rho=.75, verbose=True, device='cuda',
                 n_target_classes=9,n_iter_2_p=.22, n_iter_min_p=.06, size_decr_p=.03):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.target_class = None
        self.device = device
        self.n_target_classes = n_target_classes
        self.n_iter_2, self.n_iter_min, self.size_decr = max(int(n_iter_2_p * self.n_iter), 1), max(int(n_iter_min_p * self.n_iter), 1), max(int(size_decr_p * self.n_iter), 1)
        self.loss='md'
        self.step_size=1e-4


    def check_oscillation(self, x, j, k, y5, k3=0.5):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
          t += x[j - counter5] < x[j - counter5 - 1]
        return t <= k*k3*np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def md_loss_targeted(self, x, y, y_target):
        x_target=torch.unsqueeze(x[np.arange(x.shape[0]), y_target],1).expand(-1,x.shape[1])
        return torch.sum(nn.functional.relu(x+1e-15-x_target),dim=1)


    def smartround(self,x_adv,y,y_target):
        logits = self.model(x_adv)
        loss_indiv = self.md_loss_targeted(logits, y, y_target)
        with torch.enable_grad():
            g=torch.autograd.grad(loss_indv, [x_adv])[0].detach()
        return torch.sign(g)*0.5

    def attack_single_run(self, x_in, y_in,target_offset=None,batch_idx=None,measure=False,measure_results_dir=''):
        print  ('CGD')
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        x_upper=x.clone()+self.eps
        x_lower=x.clone()-self.eps
        x_upper=torch.clamp(x_upper, 0.0, 1.0)
        x_lower=torch.clamp(x_lower, 0.0, 1.0)
        beta_1 = .5
        beta_2 = .999
        m_t = torch.zeros_like(x)
        v_t = torch.zeros_like(x)
        m_t_over = torch.zeros_like(x)
        v_t_over = torch.zeros_like(x)
        unit=torch.arange(x.size()[0]).to(self.device)
        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone().detach()
        x_best=torch.round(x_best*255)/255.
        w1=self.eps*255/100
        output = self.model(x)
        if (target_offset is None):
            y_target = output.sort(dim=1)[1][:, -self.target_class]
        else:
            y_target=(y+target_offset) % 10
        self.n_iter=100
        w2=torch.ones_like(y)*.1
        loss_best=torch.ones_like(y)*1e6
        record=np.zeros((self.n_iter,4,x.size()[0]))
        w1_filter=1.0
        shepherd_flag=True
        for i in range(self.n_iter):
            beta_1=.1
            beta_2=.01
            if i % 15==0:
                w1/=2
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            self.eot_iter=1
            for k in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv) 
                    loss_indiv = self.md_loss_targeted(logits, y, y_target)
                    loss_over2=(nn.functional.relu(x_adv-x_upper+0*torch.sign(x_adv-x_upper))+nn.functional.relu(x_lower-x_adv+0*torch.sign(x_lower-x_adv)))
                    loss_over=torch.square(loss_over2)
                    loss_over_indv=torch.sum(loss_over,(3,2,1))
                    
                    if (i % 15 >=3):
                        maxout=torch.max(torch.max(torch.max(loss_over2,3)[0],2)[0],1)[0]
                        w2[torch.masked_select(unit, maxout>=max(.5/255,pow(2,3-i//15)*self.eps))]/=2
                    loss_per=(loss_over_indv*(1-w2))+(loss_indiv*w2)
                    loss=loss_per.sum()
                grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            grad /= float(self.eot_iter)*w1_filter
            m_t = beta_1*m_t + (1-beta_1)*grad
            v_t = beta_2*v_t + (1-beta_2)*(grad.square())
            m_cap = m_t/(1-(beta_1**(i+1)))
            v_cap = v_t/(1-(beta_2**(i+1)))
            update=(m_cap*w1)/(v_cap.sqrt()+1e-15)

            
            with torch.no_grad():
                
                x_adv = x_adv.detach()
                if (i>0):
                    x_adv = x_adv - update
                else:
                    x_adv = x_adv - torch.sign(update)*self.eps*2
                    m_t=0
                    v_t=0
                    #calibration
                    grad_mean=grad.abs().mean().item()
                    if (grad_mean>1e-2):
                        shepherd_flag=False
                    x_adv = torch.clamp(torch.min(torch.max(x_adv, x - self.eps), x + self.eps), 0.0, 1.0)

            
                x_test=torch.round(x_adv*255)/255.
                x_test = torch.clamp(torch.min(torch.max(x_test, x - self.eps), x + self.eps), 0.0, 1.0)
                pred=self.model(x_test).max(1)[1] == y_target
                x_best[pred]=x_test[pred]
            
            
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, loss))
        acc= (self.model(x_best).max(1)[1] == y_target)
        return x_best, acc, loss, x_best
   

parser = argparse.ArgumentParser(description='Targeted attacks on ImageNet')
parser.add_argument("--attack", type=str, default="CGD",help="attack to run, one of {ce, cw, dlr, md, CGD} ")
parser.add_argument("--eps", type=int,default=8, help="distance limit epsilon, either 4 or 8")
parser.add_argument("--Ninitial", type=int, default=5,help="number of choices of random initialization, a positive integer")
parser.add_argument("--Ntarget", type=int,default=5,help="number of choices of random target classes, a positive integer")
args = parser.parse_args()

method=args.attack
assert method in ['ce', 'cw', 'dlr', 'md', 'CGD'], "invalid attack"
eps=args.eps
assert (eps in [4,8]), "invalid epsilon"
Ninitial=args.Ninitial
assert Ninitial>0, "invalid number of random initializations"
Ntarget=args.Ntarget
assert (Ntarget>0), "invalid number of choices of random target classes"


band=1000
bs=10
bdiff=band/bs
ds=ImageNet('../ILSVRC')
model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
             resume_path='imagenet_linf_'+str(eps)+'.pt')#, state_dict_path='model')
model.cuda().eval()
_, test_loader = ds.make_loaders(workers=0, batch_size=bs,only_val=True)
device='cuda'
success=np.zeros((Ntarget,Ninitial,50000))

for offset_seed in range(Ntarget):
    torch.random.manual_seed(offset_seed)
    torch.cuda.random.manual_seed(offset_seed)
    target_offset=torch.floor(torch.rand(50000)*999).long()+1 
    for init_seed in range(Ninitial):
        torch.random.manual_seed(init_seed)
        torch.cuda.random.manual_seed(init_seed)

        if method in ["ce","dlr","cw","md"]:
            attack=APGDAttack_targeted(model=model,eps=eps/255,loss=method)
        if method=='CGD':
            attack=CGD(model=model,eps=eps/255)
        suc_band=np.zeros(band)
        i=0
        for im, label in iter(test_loader):
            band_offset=int(i//(band/bs))
            if (np.sum(success[offset_seed,init_seed,band_offset*band:band_offset*band+band])==0):
                print (i)
                im=im.clone().to(device)        
                perturbed,_,_,_=attack.attack_single_run(im,label.clone().to(device),target_offset=target_offset[i*bs:i*bs+bs].cuda())
                b2=(model(perturbed)).detach().max(1)[1]
                suc_batch=(((label+target_offset[i*bs:i*bs+bs]) % 1000)==b2.cpu()).detach().numpy()
            
                print (np.sum(suc_batch))
                bs_offset=int(i%(band/bs))
                suc_band[bs_offset*bs:bs_offset*bs+bs]=suc_batch
                if (bs_offset==bdiff-1):
                    success[offset_seed,init_seed,band_offset*band:band_offset*band+band]=suc_band
                    suc_band=np.zeros(band)
                    with open('record/Imagenet-'+method+str(eps),'wb') as f:
                        pickle.dump(success,f)
            i+=1
        print (np.mean(success[offset_seed,init_seed]))

