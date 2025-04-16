# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 16:43:47 2025

@author: jamie.taylor
"""


import tensorflow as tf
import numpy as np
import time 

dtype="float64"


sqrt=np.sqrt

def scheduler(epoch,lr_init,iterations):
    # alpha=1
    alpha=1
    
    lr_final=lr_init/100
    
    its0=iterations**0.5
    
    c=lr_init*((its0 - iterations)*(lr_final/lr_init)**(1/alpha)/(-1 + (lr_final/lr_init)**(1/alpha)))**alpha
    b= (-(lr_final/lr_init)**(1/alpha)*iterations + its0)/(-1 + (lr_final/lr_init)**(1/alpha))
    if epoch<its0:
        ans=lr_init
    else:

        ans = c/(b+epoch)**alpha
        
    return ans



class training():
    def __init__(self,training_objects,iterations):
        
        self.training_objects = training_objects
        
        self.h1_list = []
        self.loss_list = []
        self.epochs = []
        self.epoch=0
        
        self.a0 = 0.1
        self.b0 = 0.9
        self.eps = 10**-2
        self.iterations=iterations
        self.lr_init = 10**-2
        
        self.m= [tf.zeros_like(g) for g in training_objects.u_model.trainable_weights]
        
        self.v= [tf.zeros_like(g) for g in training_objects.u_model.trainable_weights]
        
        self.lenvars=len(self.v)
        self.r1=0.
        self.r2=0.
    
    
    def one_step(self):
        
        t0 = time.time()
        
        lr = scheduler(self.epoch,self.lr_init,self.iterations)
        print("Learning rate =",lr)
        a = 1-(1-self.a0)*lr/self.lr_init
        
        b = 1-(1-self.b0)*lr/self.lr_init
        
        loss,dloss= self.training_objects.dloss_fn()
        
        h1 = float(self.training_objects.h1_error())
        
        h1=h1*100
        self.h1_list+=[h1]
        
        print("H1 error",float(h1))
        print("Loss", float(loss))
        print("Epoch", self.epoch)
        
        self.loss_list+=[loss]
        
        self.r1 = a*self.r1+(1-a)
        self.r2 = b*self.r2+(1-b)
        
    
        for i in range(self.lenvars):
            self.m[i] = a*self.m[i]+(1-a)*dloss[i]
            self.v[i] = b*self.v[i]+(1-b)*(dloss[i]**2)
            
            hatm = self.m[i]/self.r1
            hatv = self.v[i]/self.r2
            
            update = hatm/(self.eps+hatv**0.5)
            # loss_model.layers[-1].u_model.trainable_weights[i].assign(loss_model.layers[-1].u_model.trainable_weights[i]-lr*update)
            self.training_objects.u_model.trainable_weights[i].assign_sub(lr*update)
        self.epochs+=[self.epoch]
        self.epoch+=1
        
        
        
    def train(self):
        tf.random.set_seed(1234)
        for i in range(self.iterations):
            self.one_step()
    
    
