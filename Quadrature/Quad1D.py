# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 15:28:45 2025

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np
dtype="float64"


class Quadrature_1D():
    def __init__(self,rule,N):
        self.N = N
        self.rule = rule
        
        if not self.rule=="MC":
            self.M,self.C,self.dets = self.partition_1d(self.N)

    def partition_1d(self,N):
        C = tf.constant([(2*i+1)/(2*N) for i in range(N)],dtype=dtype)
        M = tf.constant([1/(2*N) for i in range(N)],dtype=dtype)
        dets = M
        return M,C,dets
            
    
    ## 1-point Gauss rule
    def G1(self):
        
        u1 = tf.ones(self.N,dtype=dtype)
                
        w = 2*self.dets
        
        # x1 = self.M*u1+self.C
        x1 = self.C
            
        return x1,w        
    
    ## 2-point Gauss rule
    def G3(self):
        
        ones = tf.ones(self.N,dtype=dtype)
        u1 = ones/3**0.5
        u2 = -u1
        
        w = tf.concat([self.dets,self.dets],axis=-1)
        
        x1 = self.M*u1+self.C
        x2 = self.M*u2+self.C
            
        return tf.concat([x1,x2],axis=-1),w
            
    ###Vanilla Monte Carlo on [0,1]
    def MC(self):
        return tf.random.uniform([self.N],dtype=dtype),tf.ones(self.N,dtype=dtype)/self.N
    
    
    ## Grid-based Monte Carlo on [0,1]
    def P0(self):
        
        u = tf.random.uniform([self.N],dtype=dtype,minval=-1)
        x = self.M*u+self.C
        w=2*self.dets
        return x,w
    
    def P1(self):
        u = tf.random.uniform([self.N],dtype=dtype)
        x1 = self.C+self.M*u
        x2 = self.C-self.M*u
        w1 = self.dets
        w2=self.dets
        return tf.concat([x1,x2],axis=-1),tf.concat([w1,w2],axis=-1)
            
            
    def P1b(self):
        
        u1 = tf.random.uniform([self.N],dtype=dtype)
        u2 = -tf.random.uniform([self.N],dtype=dtype)
        
        x1 = self.M*u1+self.C
        x2 = self.M*u2+self.C
        
        w1 = (-2*u2/(u1 - u2))*self.dets
        w2 = (2*u1/(u1 - u2))*self.dets
        
        return tf.concat([x1,x2],axis=-1), tf.concat([w1,w2],axis=-1)
    
    def P3(self):
        
        
        u1 = tf.random.uniform([self.N],maxval=1,dtype=dtype)**(1/3)
        u2 = - u1
        u3 = tf.zeros_like(u1)
        
        w1 = self.dets*(1/(3*u1**2))
        w2 = w1
        w3 = 2*self.dets-2*w1
        
        x1 = self.M*u1+self.C
        x2 = self.M*u2+self.C
        x3 = self.M*u3+self.C
        
        return tf.concat([x1,x2,x3],axis=-1),tf.concat([w1,w2,w3],axis=-1)
    
    def __call__(self):
        if self.rule == "MC":
            return self.MC()
        elif self.rule=="P0":
            return self.P0()
        elif self.rule=="P1":
            return self.P1()
        elif self.rule=="P1b":
            return self.P1b()
        elif self.rule=="P3":
            return self.P3()
        elif self.rule=="G1":
            return self.G1()
        elif self.rule=="G3":
            return self.G3()