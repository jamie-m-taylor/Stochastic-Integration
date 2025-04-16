# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 15:32:54 2025

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np
dtype="float64"



class Quadrature_2D():
    def __init__(self,rule,N):
        self.N=N
        self.rule=rule
        
        self.R1 = tf.constant([[np.cos(np.pi*2/3),-np.sin(np.pi*2/3)],[np.sin(np.pi*2/3),np.cos(np.pi*2/3)]],dtype=dtype)
        self.R2 = tf.einsum("ij,jk->ik",self.R1,self.R1)
        
        if not rule =="MC":
            self.M,self.C,self.dets = self.partition_square(N)
    
            
    def partition_square(self,N):
        C = tf.constant([[(2*i+1)/(2*N),(2*j+1)/(2*N)] for i in range(N) for j in range(N)],dtype=dtype)
        M = tf.constant([[[1/(2*N),0],[0,1/(2*N)]] for i in range(N) for j in range(N)],dtype=dtype)
        dets = tf.linalg.det(M)
        return M,C,dets
            
    
    def R(self,t):
        return  tf.constant([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]],dtype=dtype)
    
            
    def MC(self):
        return tf.random.uniform([self.N],dtype=dtype),tf.random.uniform([self.N],dtype=dtype),tf.ones(self.N,dtype=dtype)/self.N
    
    
    
    def P0(self):
        
        u1 = tf.random.uniform([self.N**2,2],minval=-1,dtype=dtype)
        
        w1 = self.dets*4
        
        
        xy1 = tf.einsum("ijk,ik->ij",self.M,u1)+self.C
        
        return xy1[:,0],xy1[:,1],w1
    
    def P1(self):
        
        u1 = tf.random.uniform([self.N**2,2],minval=-1,dtype=dtype)
        u2 = -u1
        
        w1 = self.dets*2
        w2 = self.dets*2
        
        w= tf.concat([w1,w2],axis=-1)
        
        xy1 = tf.einsum("ijk,ik->ij",self.M,u1)+self.C
        xy2 = tf.einsum("ijk,ik->ij",self.M,u2)+self.C
        
        x = tf.concat([xy1[:,0],xy2[:,0]],axis=-1)
        y = tf.concat([xy1[:,1],xy2[:,1]],axis=-1)
        
        return x,y,w
    
            
    def P3(self):
        
        p_xi = tf.random.uniform([self.N**2],dtype=dtype)
        ind_xi = (tf.math.sign(tf.random.uniform([self.N**2],dtype=dtype)-0.5)+2)**-1
        xi = p_xi**ind_xi
        
        p_eta = tf.random.uniform([self.N**2],dtype=dtype)
        ind_eta =  (tf.math.sign(tf.random.uniform([self.N**2],dtype=dtype)-3*xi**2/(1+3*xi**2))+2)**-1
        eta = p_eta**ind_eta
        
        u1 = tf.stack([xi,eta],axis=-1)
        u2 = tf.stack([-eta,xi],axis=-1)
        u3 = tf.stack([eta,-xi],axis=-1)
        u4 = tf.stack([-xi,-eta],axis=-1)
        u5 = tf.zeros_like(u1)
        
        xy1 = tf.einsum("ijk,ik->ij",self.M,u1)+self.C
        xy2 = tf.einsum("ijk,ik->ij",self.M,u2)+self.C
        xy3 = tf.einsum("ijk,ik->ij",self.M,u3)+self.C
        xy4 = tf.einsum("ijk,ik->ij",self.M,u4)+self.C
        xy5 = tf.einsum("ijk,ik->ij",self.M,u5)+self.C
        
        w1 =self.dets*2/(3*(xi**2+eta**2))
        w2=w1
        w3=w1
        w4=w1
        w5 = 4*self.dets - 4*w1
        
        x = tf.concat([xy1[:,0],xy2[:,0],xy3[:,0],xy4[:,0],xy5[:,0]],axis=-1)
        y = tf.concat([xy1[:,1],xy2[:,1],xy3[:,1],xy4[:,1],xy5[:,1]],axis=-1)
        
        w = tf.concat([w1,w2,w3,w4,w5],axis=-1)
        return x,y,w
    
    ##2-point Gauss rule with tensor product.
    def G3(self):
        ones = tf.ones(self.N**2,dtype=dtype)
        
        w = tf.concat([self.dets]*4,axis=-1)
        
        u1 = tf.stack([ones,ones],axis=-1)*(1/(3**0.5))
        u2 = tf.stack([ones,-ones],axis=-1)*(1/(3**0.5))
        u3 = tf.stack([-ones,ones],axis=-1)*(1/(3**0.5))
        u4 = tf.stack([-ones,-ones],axis=-1)*(1/(3**0.5))
        
        xy1 = tf.einsum("ijk,ik->ij",self.M,u1)+self.C
        xy2 = tf.einsum("ijk,ik->ij",self.M,u2)+self.C
        xy3 = tf.einsum("ijk,ik->ij",self.M,u3)+self.C
        xy4 = tf.einsum("ijk,ik->ij",self.M,u4)+self.C
        
        x = tf.concat([xy1[:,0],xy2[:,0],xy3[:,0],xy4[:,0]],axis=-1)
        y = tf.concat([xy1[:,1],xy2[:,1],xy3[:,1],xy4[:,1]],axis=-1)
        
        return x,y,w
    
    def __call__(self):
        if self.rule == "MC":
            return self.MC()
        elif self.rule=="P0":
            return self.P0()
        elif self.rule=="P1":
            return self.P1()
        elif self.rule=="P3":
            return self.P3()
        elif self.rule=="G3":
            return self.G3()
