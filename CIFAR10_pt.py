import numpy as np
import time
import torch
import sys
import subprocess as sp
import torch.nn as nn
import pickle
import argparse
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import json
import math
import requests
from collections import OrderedDict


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

    def perturb(self, x_in, y_in, best_loss=False, cheap=True,target_offset=None,batch_idx=None):
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        adv = x.clone()
        acc = y==y#self.model(x).max(1)[1] == y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()
        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)
        if not cheap:
            raise ValueError('not implemented yet')
        else:

            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                    best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool,target_offset=target_offset,batch_idx=batch_idx)
                    ind_curr = (acc_curr == 0).nonzero().squeeze()
                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

        return acc.float().mean(), adv


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

    def attack_single_run_B3(self, x_in, y_in,target_offset=None,batch_idx=None,measure=False,measure_results_dir=''):
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
                    
                    if i % 15 > 4:
                        if (shepherd_flag):
                            w2[torch.masked_select(unit,torch.max(torch.max(torch.max(loss_over2,3)[0],2)[0],1)[0]>w1*2)]/=2
                        else:
                            w2[torch.masked_select(unit,torch.max(torch.max(torch.max(loss_over2,3)[0],2)[0],1)[0]>w1*5)]/=2
                    
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
   
    def perturb(self, x_in, y_in, best_loss=False, cheap=True,target_offset=None,batch_idx=None):
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        adv = x.clone()
        acc = y==y#self.model(x).max(1)[1] == y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()
        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)
        if not cheap:
            raise ValueError('not implemented yet')
        else:

            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                    best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run_B3(x_to_fool, y_to_fool,target_offset=target_offset,batch_idx=batch_idx)
                    ind_curr = (acc_curr < 2).nonzero().squeeze() #(acc_curr == 0).nonzero().squeeze()
                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
    
        return acc.float().mean(), adv


class Logger():
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()

class AutoAttack():
    def __init__(self, model, norm='Linf', eps=.3, seed=None, verbose=True,
                 attack_to_run="ceN", version='standard', is_tf_model=False,
                 device='cuda', log_path=None,APGD_rho=.75,APGD_n_iter_2_p=.22, APGD_n_iter_min_p=.06, APGD_size_decr_p=.03):

        self.model = model        
        self.norm = norm
        assert norm in ['Linf', 'L2']
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attack_to_run = attack_to_run
        self.version = version
        self.is_tf_model = is_tf_model
        self.device = device
        self.logger = Logger(log_path)
        self.APGD_rho=APGD_rho
        self.APGD_n_iter_2_p=APGD_n_iter_2_p
        self.APGD_n_iter_min_p=APGD_n_iter_min_p
        self.APGD_size_decr_p=APGD_size_decr_p

        if (attack_to_run in ["ce","dlr","cw","md"]):
            self.APGD = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=True,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,loss=attack_to_run)

        if (attack_to_run=="CGD"):
            self.CGD = CGD(self.model, n_restarts=1, n_iter=100, verbose=True,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device)

        
    def get_logits(self, x):
        if not self.is_tf_model:
            return self.model(x)
        else:
            return self.model.predict(x)
    
    def get_seed(self):
        return time.time() if self.seed is None else self.seed
    def set_seed(self,s):
        self.seed=s
    def set_APGD_param(self,APGD_rho=.75,APGD_n_iter_2_p=.22, APGD_n_iter_min_p=.06, APGD_size_decr_p=.03):
        self.APGD_rho=APGD_rho
        self.APGD_n_iter_2_p=APGD_n_iter_2_p
        self.APGD_n_iter_min_p=APGD_n_iter_min_p
        self.APGD_size_decr_p=APGD_size_decr_p
    
    def set_APGD_n_iter(self,n):
        self.apgd_targeted.n_iter = n


    def run_standard_evaluation(self, x_orig, y_orig, bs=512,target_offset=None):
        if self.verbose:
            print(self.attack_to_run)
        total=0
        targeted_success=np.zeros(x_orig.shape[0])
        untargeted_success=0
        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min( (batch_idx + 1) * bs, x_orig.shape[0])
               

                x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                y = y_orig[start_idx:end_idx].clone().to(self.device)
                correct_batch = y.eq(y)#y.eq(output.max(dim=1)[1])
                robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)

            robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                

                    
            x_adv = x_orig.clone().detach()
            startt = time.time()
            for task in range(1):
                attack=self.attack_to_run
                # item() is super important as pytorch int division uses floor rounding
                num_robust = torch.sum(robust_flags).item()

                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))

                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)
                    
                    # run attack                    
                    if attack in ["ce","dlr","cw","md"]:
                        self.APGD.seed = self.get_seed()
                        _, adv_curr = self.APGD.perturb(x, y, cheap=True,target_offset=target_offset[batch_datapoint_idcs].clone().to(self.device))

                    elif attack == 'CGD':
                        self.CGD.seed = self.get_seed()
                        _, adv_curr = self.CGD.perturb(x, y, cheap=True,target_offset=target_offset[batch_datapoint_idcs].clone().to(self.device))

                    else:
                        raise ValueError('Attack not supported')
                
                    output = self.get_logits(adv_curr)
                    false_batch = ~y.eq(output.max(dim=1)[1]).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                
                    if (target_offset is not None):
                        target_batch=output.max(dim=1)[1] == (y+target_offset[batch_datapoint_idcs].clone().to(self.device))%10
                        targeted_success[start_idx:end_idx]=target_batch.float().cpu().numpy()
                    untargeted_success+=false_batch.float().sum().item()
                    total+=list(target_batch.size())[0]
                        
        return targeted_success,untargeted_success,total
        
    def clean_accuracy(self, x_orig, y_orig, bs=512):
        n_batches = x_orig.shape[0] // bs
        acc = 0.
        for counter in range(n_batches):
            x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()
            
        if self.verbose:
            print('clean accuracy: {:.2%}'.format(acc / x_orig.shape[0]))        
        return acc.item() / x_orig.shape[0]
        




def download_gdrive(gdrive_id, fname_save):
    """ source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, fname_save):
        CHUNK_SIZE = 32768

        with open(fname_save, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    print('Download started: path={} (gdrive_id={})'.format(fname_save, gdrive_id))

    url_base = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url_base, params={'id': gdrive_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': gdrive_id, 'confirm': token}
        response = session.get(url_base, params=params, stream=True)

    save_response_content(response, fname_save)
    session.close()
    print('Download finished: path={} (gdrive_id={})'.format(fname_save, gdrive_id))


def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:  # to delete prefix 'module.' if it exists
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def load_cifar10(n_examples, data_dir='./data'):
    batch_size = 100
    transform_chain = transforms.Compose([transforms.ToTensor()])
    item = datasets.CIFAR10(root=data_dir, train=False, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=batch_size, shuffle=False, num_workers=0)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if batch_size * i >= n_examples:
            break

    x_test = torch.cat(x_test)[:n_examples]
    y_test = torch.cat(y_test)[:n_examples]

    return x_test, y_test

def load_cifar10_train(n_examples, data_dir='./data'):
    batch_size = 100
    transform_chain = transforms.Compose([transforms.ToTensor()])
    item = datasets.CIFAR10(root=data_dir, train=True, transform=transform_chain, download=True)
    train_loader = data.DataLoader(item, batch_size=batch_size, shuffle=False, num_workers=0)

    x_train, y_train = [], []
    for i, (x, y) in enumerate(train_loader):
        x_train.append(x)
        y_train.append(y)
        if batch_size * i >= n_examples:
            break

    x_train = torch.cat(x_train)[:n_examples]
    y_train = torch.cat(y_train)[:n_examples]

    return x_train, y_train
def load_model(model_name, model_dir='./models', norm='Linf'):
    from model_zoo.models import model_dicts as all_models
    model_dir += '/{}'.format(norm)
    model_path = '{}/{}.pt'.format(model_dir, model_name)
    model_dicts = all_models[norm]
    if not isinstance(model_dicts[model_name]['gdrive_id'], list):
        model = model_dicts[model_name]['model']()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.isfile(model_path):
            download_gdrive(model_dicts[model_name]['gdrive_id'], model_path)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # needed for the model of `Carmon2019Unlabeled`
        try:
            state_dict = rm_substr_from_state_dict(checkpoint['state_dict'], 'module.')
        except:
            state_dict = rm_substr_from_state_dict(checkpoint, 'module.')

        model.load_state_dict(state_dict,strict=False)
        return model.eval()

    # If we have an ensemble of models (e.g., Chen2020Adversarial)
    else:
        model = model_dicts[model_name]['model']()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        for i, gid in enumerate(model_dicts[model_name]['gdrive_id']):
            if not os.path.isfile('{}_m{}.pt'.format(model_path, i)):
                download_gdrive(gid, '{}_m{}.pt'.format(model_path, i))
            checkpoint = torch.load('{}_m{}.pt'.format(model_path, i), map_location=torch.device('cpu'))
            try:
                state_dict = rm_substr_from_state_dict(checkpoint['state_dict'], 'module.')
            except:
                state_dict = rm_substr_from_state_dict(checkpoint, 'module.')
            model.models[i].load_state_dict(state_dict, strict=False)
            model.models[i].eval()
        return model.eval()


x, y = load_cifar10(n_examples=10000)

start=time.time()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

model_names=['Ding2020MMA','Wong2020Fast','Hendrycks2019Using','Wang2020Improving','Sehwag2020Hydra','Carmon2019Unlabeled','Wu2020Adversarial_extra']
modelnumber=len(model_names)

parser = argparse.ArgumentParser(description='Targeted attacks on CIFAR10')
parser.add_argument("--attack", type=str, default="CGD",help="attack to run, one of {ce, cw, dlr, md, CGD} ")
parser.add_argument("--eps", type=int,default=16, help="distance limit epsilon, an integer in [1,255]")
parser.add_argument("--Ninitial", type=int, default=20,help="number of choices of random initialization, a positive integer")
parser.add_argument("--Nimages", type=int,default=10000,help="number of images to perturbed, an positive integer no larger than 10,000")
args = parser.parse_args()


attack_to_run=args.attack
assert attack_to_run in ['ce', 'cw', 'dlr', 'md', 'CGD'], "invalid attack"
eps=float(args.eps)/255.
assert (0<args.eps and 256>args.eps), "invalid epsilon"
seednumber=args.Ninitial
assert seednumber>0, "invalid number of random initializations"
samples=args.Nimages
assert (0<samples and 10000>=samples), "invalid number of images"
x=x[:samples]
y=y[:samples]

torch.random.manual_seed(0)
torch.cuda.random.manual_seed(0)
target_offset=torch.floor(torch.rand(samples)*9).long()+1


record=np.zeros((seednumber,modelnumber,samples))
for seed in range(seednumber):
  for modelindex in range (modelnumber):
    model = load_model(model_names[modelindex], norm='Linf')
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    adversary = AutoAttack(model, norm='Linf', eps=eps, version='custom', attack_to_run=attack_to_run)
    adversary.set_seed(seed)
    targeted_success,untargeted_success,total = adversary.run_standard_evaluation(x, y,bs=512,target_offset=target_offset.cuda())
    record[seed][modelindex]=targeted_success
    with open('CIFAR10-eps'+str(int(eps*255))+'-'+attack_to_run+'','wb') as f:
        pickle.dump(record,f)
    del model
    print (np.mean(targeted_success))
