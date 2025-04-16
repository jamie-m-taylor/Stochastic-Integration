# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 15:52:46 2025

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np
dtype="float64"

class Quadrature_3D():
    def __init__(self,rule,N):
        self.rule=rule
        self.N=N
        
        if not rule=="MC":
            self.M,self.C,self.dets = self.partition_cube(N)
            
    
    def partition_cube(self,N):
        C = tf.constant([[(2*i+1)/(2*N),(2*j+1)/(2*N),(2*k+1)/(2*N)] for i in range(N) for j in range(N) for k in range(N)],dtype=dtype)
        M = tf.constant([[[1/(2*N),0,0],[0,1/(2*N),0],[0,0,1/(2*N)]] for i in range(N) for j in range(N) for k in range(N)],dtype=dtype)
        dets = tf.linalg.det(M)
        return M,C,dets
    
    def MC(self):
        return tf.random.uniform([self.N],dtype=dtype),tf.random.uniform([self.N],dtype=dtype),tf.random.uniform([self.N],dtype=dtype),tf.ones(self.N,dtype=dtype)/self.N

            
    def P0(self):
        u1 = tf.random.uniform([self.N**3,3],minval=-1,dtype=dtype)
        
        w1 = self.dets*8
        
        
        xy1 = tf.einsum("ijk,ik->ij",self.M,u1)+self.C
        
        return xy1[:,0],xy1[:,1],xy1[:,2],w1
    
            
    def P1(self):
        
        u1 = tf.random.uniform([self.N**3,3],minval=-1,dtype=dtype)
        u2 = -u1
        w1 = self.dets*4
        w = tf.concat([w1,w1],axis=-1)
        
        xyz1 = tf.einsum("ijk,ik->ij",self.M,u1)+self.C
        xyz2 = tf.einsum("ijk,ik->ij",self.M,u2)+self.C
        
        x = tf.concat([xyz1[:,0],xyz2[:,0]],axis=-1)
        y = tf.concat([xyz1[:,1],xyz2[:,1]],axis=-1)
        z = tf.concat([xyz1[:,2],xyz2[:,2]],axis=-1)
        
        return x,y,z,w
            
            
    def P3(self):
            
        
        u1 = tf.random.uniform([self.N**3],maxval=1,dtype=dtype)**(1/3)
        u2 = - u1
        u3 = tf.zeros_like(u1)
        
        
        w1 = (1/(3*u1**2))
        w2 = w1
        w3 = (2-2*w1)
        
        
        pz = tf.stack([u1]*5+[u2]*5+[u3]*5,axis=-1)
        pwz = tf.stack([w1]*5+[w2]*5+[w3]*5,axis=-1)
        
        p_xi = tf.random.uniform([self.N**3],dtype=dtype)
        ind_xi = (tf.math.sign(tf.random.uniform([self.N**3],dtype=dtype)-0.5)+2)**-1
        xi = p_xi**ind_xi
        
        p_eta = tf.random.uniform([self.N**3],dtype=dtype)
        ind_eta =  (tf.math.sign(tf.random.uniform([self.N**3],dtype=dtype)-3*xi**2/(1+3*xi**2))+2)**-1
        eta = p_eta**ind_eta
        
        u1 = tf.stack([xi,eta],axis=-1)
        u2 = tf.stack([-eta,xi],axis=-1)
        u3 = tf.stack([eta,-xi],axis=-1)
        u4 = tf.stack([-xi,-eta],axis=-1)
        u5 = tf.zeros_like(u1)
        
        
        w1 =2/(3*(xi**2+eta**2))
        w2=w1
        w3=w1
        w4=w1
        w5 = (4 - 4*w1)
        
        px = tf.stack([u1[:,0],u2[:,0],u3[:,0],u4[:,0],u5[:,0]]*3,axis=-1)
        py = tf.stack([u1[:,1],u2[:,1],u3[:,1],u4[:,1],u5[:,1]]*3,axis=-1)
        
        
        pwxy = tf.stack([w1,w2,w3,w4,w5]*3,axis=-1)
        
        Uxyz = tf.stack([pz,px,py],axis=-1)
        
        xyz = tf.einsum("mij->imj",tf.einsum("ijk,imk->mij",self.M,Uxyz)+self.C)
        
        x = tf.reshape(xyz[:,:,0],[self.N**3*15])
        y = tf.reshape(xyz[:,:,1],[self.N**3*15])
        z = tf.reshape(xyz[:,:,2],[self.N**3*15])
        
        w = tf.reshape(tf.einsum("i,ij->ij",self.dets,pwz*pwxy),[self.N**3*15])
        
        return x,y,z,w
    
    
    
    def G3(self):
        ones = tf.ones(self.N**3,dtype=dtype)
        
        w = tf.concat([self.dets]*8,axis=-1)
        
        u1 = tf.stack([ones,ones,ones],axis=-1)*(1/(3**0.5))
        u2 = tf.stack([-ones,ones,ones],axis=-1)*(1/(3**0.5))
        u3 = tf.stack([ones,-ones,ones],axis=-1)*(1/(3**0.5))
        u4 = tf.stack([ones,ones,-ones],axis=-1)*(1/(3**0.5))
        u5 = tf.stack([ones,-ones,-ones],axis=-1)*(1/(3**0.5))
        u6 = tf.stack([-ones,ones,-ones],axis=-1)*(1/(3**0.5))
        u7 = tf.stack([-ones,-ones,ones],axis=-1)*(1/(3**0.5))
        u8 = tf.stack([-ones,-ones,-ones],axis=-1)*(1/(3**0.5))
        
        xy1 = tf.einsum("ijk,ik->ij",self.M,u1)+self.C
        xy2 = tf.einsum("ijk,ik->ij",self.M,u2)+self.C
        xy3 = tf.einsum("ijk,ik->ij",self.M,u3)+self.C
        xy4 = tf.einsum("ijk,ik->ij",self.M,u4)+self.C
        xy5 = tf.einsum("ijk,ik->ij",self.M,u5)+self.C
        xy6 = tf.einsum("ijk,ik->ij",self.M,u6)+self.C
        xy7 = tf.einsum("ijk,ik->ij",self.M,u7)+self.C
        xy8 = tf.einsum("ijk,ik->ij",self.M,u8)+self.C
        
        
        x = tf.concat([xy1[:,0],xy2[:,0],xy3[:,0],xy4[:,0],xy5[:,0],xy6[:,0],xy7[:,0],xy8[:,0]],axis=-1)
        y = tf.concat([xy1[:,1],xy2[:,1],xy3[:,1],xy4[:,1],xy5[:,1],xy6[:,1],xy7[:,1],xy8[:,1]],axis=-1)
        z = tf.concat([xy1[:,2],xy2[:,2],xy3[:,2],xy4[:,2],xy5[:,2],xy6[:,2],xy7[:,2],xy8[:,2]],axis=-1)
        return x,y,z,w
    
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




        