# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:13:06 2025

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np
import os 
import csv

dtype="float64"

class Quad_triangles():
    def __init__(self,rule,N):
        
        if rule=="P2t" or rule=="P1t":
            self.R1 = tf.constant([[np.cos(np.pi*2/3),-np.sin(np.pi*2/3)],[np.sin(np.pi*2/3),np.cos(np.pi*2/3)]],dtype=dtype)
            self.R2 = tf.einsum("ij,jk->ik",self.R1,self.R1)
            
        if rule=="P2t":         
            with open(os.path.join("Quadrature","QuadData","P2_triangle")+'.csv', 'r') as f:
                reader = csv.reader(f)
                data = list(reader)
            
                
            self.xyP2 = tf.constant(np.array(data,dtype=float),dtype=dtype)
        
        self.M,self.C,self.dets = self.partition_triangle(N)
        self.N=N
        self.rule = rule
        
        

    
    
    def R(self,t):
        return  tf.constant([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]],dtype=dtype)
            
    def equal_to_right(self,upper=True):
        m1 = tf.constant([[1,0],[0,3**0.5/3]],dtype=dtype)/(6**0.5/2)
        c1 = tf.einsum("ij,j->i",m1,tf.constant([0,1],dtype=dtype))
        if upper:
            R4 = self.R(np.pi/4)
        else:
            R4 = self.R(np.pi*(1/4+1))
        c1 = tf.einsum("ij,j->i",R4,c1) 
        if upper:
            c1-=tf.constant([1,1],dtype=dtype)
        M = tf.einsum("ij,jk->ik",R4,m1)
        return M,c1

    def partition_triangle(self,n):
        h=1/n
        m_l,c_l = self.equal_to_right(upper=False)
        m_u,c_u = self.equal_to_right(upper=True)
        x_test =tf.constant([i*h for i in range(n)],dtype=dtype)
        px,py = tf.meshgrid(x_test,x_test)
        px = tf.reshape(px,[n**2])
        py = tf.reshape(py,[n**2])
        X = tf.stack([px,py],axis=-1)
        M = tf.stack([m_l*h]*(n**2)+[m_u*h]*(n**2),axis=0)
        C = tf.concat([X-c_l*h,X-c_u*h],axis=0)
        return M,C,tf.linalg.det(M)
    
    
    def sample_rows(self,tensor, M):
        N = tf.shape(tensor)[0]
        indices = tf.random.uniform([M], minval=0, maxval=N, dtype=tf.int32)
        sampled_tensor = tf.gather(tensor, indices)
        return sampled_tensor
    
    def sample_triangle_unif(self,n):
        h=3**0.5/2
        V = tf.constant([[-0.5,-h],[-0.5,h],[0,0.1]],dtype=dtype)
        U = tf.sort(tf.random.uniform([n,2],dtype=dtype),axis=-1)
        l1 = U[:,0]
        l2 = U[:,1]-U[:,0]
        l3 = 1-U[:,1]
        L = tf.stack([l1,l2,l3],axis=-1)
        return tf.einsum("ij,jk->ik",L,V)
    
    
    def P0t(self):
        X0 = self.sample_triangle_unif(len(self.C))
        
        X0 = self.C +tf.einsum("kij,kj->ki",self.M,X0)
        
        X = X0[:,0]
        Y = X0[:,1]
        
        W = 3*(3**0.5)/(4)*self.dets
        return X,Y,W
    
    def P1t(self):
        X0 = self.sample_triangle_unif(len(self.C))
        X1 = tf.einsum("ij,kj->ki",self.R1,X0)
        X2 = tf.einsum("ij,kj->ki",self.R2,X0)
        
        X0 = self.C +tf.einsum("kij,kj->ki",self.M,X0)
        X1 = self.C +tf.einsum("kij,kj->ki",self.M,X1)
        X2 = self.C +tf.einsum("kij,kj->ki",self.M,X2)
        
        X = tf.concat([X0[:,0],X1[:,0],X2[:,0]],axis=-1)
        Y = tf.concat([X0[:,1],X1[:,1],X2[:,1]],axis=-1)
        
        W0 = 3*(3**0.5)/(4)*self.dets
        W = tf.concat([W0,W0,W0],axis=-1)/3
        return X,Y,W
        
    def P2t(self):
        
        # X0 = tf.random.shuffle(xylist)[:len(C0)]
        X0 =self.sample_rows(self.xyP2,len(self.C))
        
        X1 = tf.einsum("ij,kj->ki",self.R1,X0)
        X2 = tf.einsum("ij,kj->ki",self.R2,X0)
        X3 = 0*X0
        
        W1 = (3**0.5)/(16*tf.reduce_sum(X0**2,axis=-1))
        
        X0 = self.C +tf.einsum("kij,kj->ki",self.M,X0)
        X1 = self.C +tf.einsum("kij,kj->ki",self.M,X1)
        X2 = self.C +tf.einsum("kij,kj->ki",self.M,X2)
        X3 = self.C +tf.einsum("kij,kj->ki",self.M,X3)
        
        X = tf.concat([X0[:,0],X1[:,0],X2[:,0],X3[:,0]],axis=-1)
        Y = tf.concat([X0[:,1],X1[:,1],X2[:,1],X3[:,1]],axis=-1)
        
        W0 = 3*(3**0.5)/(4)*self.dets
        W1 = W1*self.dets
        W = tf.concat([W1,W1,W1,W0-3*W1],axis=-1)
        return X,Y,W
    
    def __call__(self):
        if self.rule=="P1t":
            return self.P1t()
        elif self.rule=="P2t":
            return self.P2t()
        elif self.rule=="P0t":
            return self.P0t