# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:58:10 2025

@author: jamie.taylor
"""


import tensorflow as tf
import numpy as np
import os 
import csv

dtype="float64"

class Quad_tetrahedra():
    def __init__(self,rule,N):
        
        self.rule = rule
        
        p1 = tf.constant([1,1,1],dtype=dtype)
        p2 = tf.constant([-1,-1,1],dtype=dtype)
        p3 = tf.constant([-1,1,-1],dtype=dtype)
        p4 = tf.constant([1,-1,-1],dtype=dtype)
        
        M1,C1=self.totet(p2,p3,p4,-p1)
        
        M2,C2=self.totet(-p2,p3,p4,p1)
        
        M3,C3=self.totet(p2,-p3,p4,p1)
        
        M4,C4=self.totet(p2,p3,-p4,p1)
        
        
        self.m_ref=tf.stack([tf.eye(3,dtype=dtype),M1,M2,M3,M4],axis=0)
        self.c_ref = tf.stack([tf.zeros([3],dtype=dtype),C1,C2,C3,C4],axis=0)
        
        if type(N)==int:
            self.M,self.C,self.dets = self.partition_uniform(N)
        else:
            self.M,self.C,self.dets = self.partition_nonuniform(N[0],N[1],N[2])
        
        if rule=="P1t":
            self.R0 = tf.constant([[0,0,-1],[0,-1,0],[1,0,0]],dtype=dtype)
        
        if rule=="P2t":         
            
            with open(os.path.join("Quadrature","QuadData","P2_tetrahedron")+'.csv', 'r') as f:
                reader = csv.reader(f)
                data = list(reader)
            
            self.xyzP2 = tf.constant(np.array(data,dtype=float),dtype=dtype)
            R1 = tf.constant([[0,1,0],[0,0,1],[1,0,0]],dtype=dtype)
            R2 = tf.constant([[-1,0,0],[0,1,0],[0,0,-1]],dtype=dtype)

            self.Rlist = [tf.constant([[1,0,0],[0,1,0],[0,0,1]],dtype=dtype),
                     R1,
                     tf.matmul(R1,R1),
                     R2,
                     tf.matmul(R1,R2),
                     tf.matmul(R2,R1),
                     tf.matmul(tf.matmul(R1,R1),R2),
                     tf.matmul(R2,tf.matmul(R1,R1)),
                     tf.matmul(R1,tf.matmul(R2,tf.matmul(R1,R1))),
                     tf.matmul(R1,tf.matmul(R1,tf.matmul(R2,R1))),
                     tf.matmul(R2,tf.matmul(R1,R2)),
                     tf.matmul(R2,tf.matmul(R1,tf.matmul(R1,R2)))
                     ]
            
            
        
    def sample_tetrahedron_unif(self,n):
        V = tf.constant([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],dtype=dtype)
        U = tf.sort(tf.random.uniform([n,3],dtype=dtype),axis=-1)
        l1 = U[:,0]
        l2 = U[:,1]-U[:,0]
        l3 = U[:,2]-U[:,1]
        l4 = 1-U[:,2]
        L = tf.stack([l1,l2,l3,l4],axis=-1)
        return tf.einsum("ij,jk->ik",L,V)
    
    def sample_rows(self,tensor, M):
        N = tf.shape(tensor)[0]
        indices = tf.random.uniform([M], minval=0, maxval=N, dtype=tf.int32)
        sampled_tensor = tf.gather(tensor, indices)
        return sampled_tensor
    
    def totet(self,v1,v2,v3,v4):
        p1 = tf.constant([1,1,1],dtype=dtype)
        p2 = tf.constant([-1,-1,1],dtype=dtype)
        p3 = tf.constant([-1,1,-1],dtype=dtype)
        p4 = tf.constant([1,-1,-1],dtype=dtype)
        f1 = v2-v1
        f2 = v3-v1
        f3 = v4-v1
        e1 = p2-p1
        e2 = p3-p1
        e3 = p4-p1
        F = tf.stack([f1,f2,f3],axis=-1)
        E = tf.stack([e1,e2,e3],axis=-2)
        A = tf.einsum("ij,jk->ik",F,tf.linalg.inv(E))
        b = v1-tf.einsum("ij,j->i",A,p1)
        return A,b
    
    def partition_uniform(self,n):
        h=1/(2*n)
        
        x_test =tf.constant([i*h*2 for i in range(n)],dtype=dtype)
        px,py,pz = tf.meshgrid(x_test,x_test,x_test)
        px = tf.reshape(px,[n**3])
        py = tf.reshape(py,[n**3])
        pz = tf.reshape(pz,[n**3])
        
        xyz = tf.stack([px,py,pz],axis=-1)
        
        M = tf.stack([self.m_ref]*(n**3),axis=0)*h
        
        pC = tf.einsum("ijk->jik",tf.stack([self.c_ref]*(n**3),axis=0))+tf.constant([1,1,1],dtype=dtype)
        
        C = tf.einsum("jik->ijk",h*pC+xyz)
        
        
        C = tf.stack([C[i,j] for i in range(n**3) for j in range(len(self.c_ref))],axis=0)
        M = tf.stack([M[i,j] for i in range(n**3) for j in range(len(self.m_ref))],axis=0)
        
        return M,C,tf.linalg.det(M)
    
    def partition_nonuniform(self,nx,ny,nz):
        hx=1/(2*nx)
        hy=1/(2*ny)
        hz=1/(2*nz)
        
        x_test =tf.constant([i*hx*2 for i in range(nx)],dtype=dtype)
        y_test =tf.constant([i*hy*2 for i in range(ny)],dtype=dtype)
        z_test =tf.constant([i*hz*2 for i in range(nz)],dtype=dtype)
        px,py,pz = tf.meshgrid(x_test,y_test,z_test)
        px = tf.reshape(px,[nx*ny*nz])
        py = tf.reshape(py,[nx*ny*nz])
        pz = tf.reshape(pz,[nx*ny*nz])
        
        xyz = tf.stack([px,py,pz],axis=-1)
        
        A_scale = tf.constant([[hx,0,0],[0,hy,0],[0,0,hz]],dtype=dtype)
        
        pM = tf.stack([self.m_ref]*(nx*ny*nz),axis=0)
        M=tf.einsum("ijkl,mk->ijml",pM,A_scale)
        
        pC = tf.einsum("ijk->jik",tf.stack([self.c_ref]*(nx*ny*nz),axis=0))+tf.constant([1,1,1],dtype=dtype)
        
        C = tf.einsum("jik->ijk",tf.einsum("ijk,kl->ijl",pC,A_scale)+xyz)
        
        
        C = tf.stack([C[i,j] for i in range(nx*ny*nz) for j in range(len(self.m_ref))],axis=0)
        M = tf.stack([M[i,j] for i in range(nx*ny*nz) for j in range(len(self.m_ref))],axis=0)
        
        return M,C,tf.linalg.det(M)
    
    
            
    def P1(self):
    
        # Sample rows from xyzlist
        X0 = self.sample_tetrahedron_unif(len(self.C))
        
        # Apply successive transformations
        X_transformed = [X0]
        for _ in range(3):
            X_transformed.append(tf.einsum("ij,kj->ki", self.R0, X_transformed[-1]))
        
        # Apply the final transformation with M0 and C0
        X_all = [self.C + tf.einsum("kij,kj->ki", self.M, X) for X in X_transformed]
    
        # Concatenate X, Y, Z coordinates
        X = tf.concat([X[:, 0] for X in X_all], axis=-1)
        Y = tf.concat([X[:, 1] for X in X_all], axis=-1)
        Z = tf.concat([X[:, 2] for X in X_all], axis=-1)
        
        # Concatenate weights
        W0 = (2/3) * self.dets
        W = tf.concat([W0] * 4, axis=-1)
        
        return X, Y, Z, W
    
    def P2(self):


        # Sample M rows from xyzlist
        X0 = self.sample_rows(self.xyzP2, len(self.C))
        W1 = 2 / (15 * tf.reduce_sum(X0**2, axis=-1)) * self.dets

        # Apply transformations
        X_transformed = [tf.einsum("ij,kj->ki", R, X0) for R in self.Rlist]
        
        # Calculate X_all with the first being C0 with zero transformation
        X_all = [self.C + tf.einsum("kij,kj->ki", self.M, X0) * 0] + \
                [self.C + tf.einsum("kij,kj->ki", self.M, X) for X in X_transformed]
        
        # Concatenate X, Y, Z coordinates
        X = tf.concat([X[:, 0] for X in X_all], axis=-1)
        Y = tf.concat([X[:, 1] for X in X_all], axis=-1)
        Z = tf.concat([X[:, 2] for X in X_all], axis=-1)
        
        # Concatenate weights
        W = tf.concat([8/3 * self.dets - 12 * W1] + [W1] * 12, axis=-1)
        
        return X, Y, Z, W
    
    def __call__(self):
        if self.rule=="P1t":
            return self.P1()
        elif self.rule=="P2t":
            return self.P2()
    