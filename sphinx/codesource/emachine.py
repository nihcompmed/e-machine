#!/usr/bin/env python
# coding: utf-8

import numpy as np
#import numpy.linalg as nplin

#=========================================================================================
# convert s[n_seq,n_var] to ops[n_seq,n_ops]
def operators(s):
    #generate terms in the energy function
    n_seq,n_var = s.shape
    ops = np.zeros((n_seq,n_var+int(n_var*(n_var-1)/2.0)))

    jindex = 0
    for index in range(n_var):
        ops[:,jindex] = s[:,index]
        jindex +=1

    for index in range(n_var-1):
        for index1 in range(index+1,n_var):
            ops[:,jindex] = s[:,index]*s[:,index1]
            jindex +=1
            
    return ops
#=========================================================================================
def energy_ops(ops,w):
    return np.sum(ops*w[np.newaxis,:],axis=1)
#=========================================================================================
# 
def generate_seq(n_var,n_seq,n_sample=30,g=1.0):
    n_ops = n_var+int(n_var*(n_var-1)/2.0)
    #w_true = g*(np.random.rand(ops.shape[1])-0.5)/np.sqrt(float(n_var))
    w_true = np.random.normal(0.,g/np.sqrt(n_var),size=n_ops)
    
    samples = np.random.choice([1.0,-1.0],size=(n_seq*n_sample,n_var),replace=True)
    ops = operators(samples)

    #sample_energy = energy_ops(ops,w_true)
    sample_energy = ops.dot(w_true)

    p = np.exp(sample_energy)
    p /= np.sum(p)
    out_samples = np.random.choice(np.arange(n_seq*n_sample),size=n_seq,replace=True,p=p)
    
    return w_true,samples[out_samples]
#=========================================================================================
# find coupling w from sequences s
# input: ops[n_seq,n_ops]
# output: w[n_ops], E_av
def fit(ops,eps=0.1,max_iter=151,alpha=0.1):
    E_av = np.zeros(max_iter)
    n_ops = ops.shape[1]
    cov_inv = np.eye(ops.shape[1])

    np.random.seed(13)
    w = np.random.rand(n_ops)-0.5    
    for i in range(max_iter):
        #if eps_type == 'random':
        #    eps_scale = np.random.rand()/np.max([1.,np.max(np.abs(w))])
                                 
        #energies_w = energy_ops(ops,w)
        energies_w = ops.dot(w)

        energies_max = energies_w.max()  
        probs_w = np.exp((energies_w-energies_max)*(eps-1)) # to avoid a too lager value
        z_data = np.sum(probs_w)
        probs_w /= z_data
        ops_expect_w = np.sum(probs_w[:,np.newaxis]*ops,axis=0)

        E_av[i] = energies_w.mean()
        w += alpha*cov_inv.dot((ops_expect_w - w*eps))        
              
    return w,-E_av[-1]
#=========================================================================================
def hopfield_method(s):
    ops = operators(s)
    w = np.mean(ops,axis=0)
    #print('hopfield error ',nplin.norm(w-w_true))
    return w
#=========================================================================================    
def MLE_method(seq,max_iter=150,alpha=5e-2):
    import itertools
    n_seq,n_var = seq.shape
    
    seq_all = np.asarray(list(itertools.product([1.0, -1.0], repeat=n_var)))
    #print('all configs size:',seq_all.shape)

    ops = operators(seq)    
    cov_inv = np.eye(ops.shape[1])
    ops_obs = np.mean(ops,axis=0)
    ops_model = operators(seq_all)

    n_ops = ops.shape[1]
    
    np.random.seed(13)
    w = np.random.rand(n_ops)-0.5    
    for iterate in range(max_iter):
        energies_w = energy_ops(ops_model,w)
        probs_w = np.exp(energies_w)
        probs_w /= np.sum(probs_w)
            
        w += alpha*cov_inv.dot(ops_obs - np.sum(ops_model*probs_w[:,np.newaxis],axis=0))

    #print('final',iterate,MSE)

    return w  
 
#========================================================================================= 
def PLE_method(seqs):
#https://github.com/eltrompetero/coniii
    import coniii

    n_var = seqs.shape[1]
    n_ops = n_var+int(n_var*(n_var-1)/2.0)
    
    ## pseudo likelihood estimation
    np.random.seed(13)
    # Define common functions
    calc_e,calc_observables,mchApproximation = coniii.define_ising_helper_functions()      
    get_multipliers_r,calc_observables_r = coniii.define_pseudo_ising_helper_functions(n_var)

    solver = coniii.Pseudo(n_var,calc_observables=calc_observables,
                    calc_observables_r=calc_observables_r,
                    get_multipliers_r=get_multipliers_r)

    w = solver.solve(seqs,np.zeros(n_ops))
        
    return w
