# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 16:18:19 2025

@author: jamie.taylor
"""


import tensorflow as tf
import numpy as np

from Quadrature.Quadratures import Quadrature

dtype="float64"


# if dim==3:
#     nval = 7

class training_objects_1D():
    def __init__(self,u_model,quad_rule):
        self.u_model=u_model
        self.quad_rule=quad_rule
    
        nval = 500
        self.h1_exact = 2.777476908
        
        self.fine_quad = Quadrature(1,"G3",nval)
            
    
    def u_exact(self,x):
        return ((x - 1/2)**2 - 1/16)*tf.math.sin(np.pi*x)*10
    
    
    def rhs(self,x):
        return (((32 + (-16*x**2 + 16*x - 3)*np.pi**2)*tf.math.sin(np.pi*x))/16 + 4*np.pi*(x - 1/2)*tf.math.cos(np.pi*x))*10
    

    @tf.function
    def loss_function(self):
        x,w = self.quad_rule()
        with tf.GradientTape() as t2:
            t2.watch(x)
            v=tf.squeeze(self.u_model(x))
        dv = t2.gradient(v,x)
        return tf.reduce_sum((0.5*dv**2+self.rhs(x)*v)*w)
    
    
    @tf.function
    def loss_at_exact(self):
        x,w = self.quad_rule()
        with tf.GradientTape() as t2:
            t2.watch(x)
            v=tf.squeeze(self.u_exact(x))
        dv = t2.gradient(v,x)
        return tf.reduce_sum((0.5*dv**2+self.rhs(x)*v)*w)
    
    @tf.function
    def dloss_fn(self):
        with tf.GradientTape() as t1:
            loss = self.loss_function()
        dloss = t1.gradient(loss,self.u_model.trainable_weights)
        return loss,dloss 
    
    @tf.function
    def h1_error(self):
        x,w = self.fine_quad()
        with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as t1:
            v = tf.squeeze(self.u_model(x))-tf.squeeze(self.u_exact(x))
        dv = t1.jvp(v)
        
        h1 = tf.reduce_sum(dv**2*w)**0.5
        
        return h1/self.h1_exact
    
    def estimate_var_loss(self,counts):
        losslist = [self.loss_function() for i in range(counts)]
        return np.var(losslist)
    def estimate_var_exact(self,counts):
        losslist = [self.loss_at_exact() for i in range(counts)]
        return np.var(losslist)
    

class training_objects_2D():
    def __init__(self,u_model,quad_rule):
        self.u_model=u_model
        self.quad_rule=quad_rule
    
        nval = 16
        self.h1_exact = 2.831984948
        
        
        self.fine_quad = Quadrature(2,"G3",nval)
            
        
        
    def u_exact(self,xy):
        x,y=tf.unstack(xy,axis=-1)
        return (tf.math.sin(np.pi*x)*tf.math.sin(np.pi*y)*((x - 2**(-1))**2 + (y - 2**(-1))**2 - 4**(-2)))*10
    
    def rhs(self,xy):
        x,y=tf.unstack(xy,axis=-1)
        return 5*((32 + (-16*x**2 - 16*y**2 + 16*x + 16*y - 7)*np.pi**2)*tf.math.sin(np.pi*y) + 32*np.pi*(y - 1/2)*tf.math.cos(np.pi*y))*tf.math.sin(np.pi*x)/4 + 40*tf.math.cos(np.pi*x)*np.pi*tf.math.sin(np.pi*y)*(x - 1/2)
    
    @tf.function
    def loss_function(self):
        x,y,w =self.quad_rule()
        
        with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as tx:
            xy = tf.stack([x,y],axis=-1)
            v = (self.u_model(xy))
        dvx = tx.jvp(v)
        
        
        with tf.autodiff.ForwardAccumulator(y, tf.ones_like(y)) as ty:
            xy = tf.stack([x,y],axis=-1)
            v = (self.u_model(xy))
        dvy = ty.jvp(v)
        
        RHS = self.rhs(xy)
        
        
        integrand =   (dvx**2/2+dvy**2/2) +v*RHS
        return tf.reduce_sum(integrand*w)
    
    
    @tf.function
    def loss_at_exact(self):
        x,y,w =self.quad_rule()
        
        with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as tx:
            xy = tf.stack([x,y],axis=-1)
            v = (self.u_exact(xy))
        dvx = tx.jvp(v)
        
        
        with tf.autodiff.ForwardAccumulator(y, tf.ones_like(y)) as ty:
            xy = tf.stack([x,y],axis=-1)
            v = (self.u_exact(xy))
        dvy = ty.jvp(v)
        
        RHS = self.rhs(xy)
        
        
        integrand =   (dvx**2/2+dvy**2/2) +v*RHS
        return tf.reduce_sum(integrand*w)
    
    @tf.function
    def dloss_fn(self):
        with tf.GradientTape() as t1:
            loss = self.loss_function()
        dloss = t1.gradient(loss,self.u_model.trainable_weights)
        return loss,dloss 
    
    @tf.function
    def h1_error(self):
        x,y,w = self.fine_quad()
        with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as tx:
            with tf.autodiff.ForwardAccumulator(y, tf.ones_like(y)) as ty:
                xy = tf.stack([x,y],axis=-1)
                ue = tf.squeeze(self.u_exact(xy)-self.u_model(xy))
            due_y = ty.jvp(ue)
        due_x = tx.jvp(ue)
        
        return tf.reduce_sum(((due_x)**2+(due_y)**2)*w)**0.5
    
    def estimate_var_loss(self,counts):
        losslist = [self.loss_function() for i in range(counts)]
        return np.var(losslist)
    
    def estimate_var_exact(self,counts):
        losslist = [self.loss_at_exact() for i in range(counts)]
        return np.var(losslist)
    
        
    def estimate_var_grad(self,counts):
        loss,dloss1 = self.dloss_fn()
        Ex = tf.zeros_like(tf.concat([tf.reshape(g,[-1]) for g in dloss1],-1))
        Ex2 = tf.einsum("i,j->ij",Ex,Ex)
        
        for i in range(counts):
            loss,dloss = self.dloss_fn()
            dloss = tf.concat([tf.reshape(g,[-1]) for g in dloss],-1)
            Ex2 += tf.einsum("i,j->ij",dloss,dloss)/counts
            Ex += dloss/counts
        cov = Ex2-tf.einsum("i,j->ij",Ex,Ex)
        trace_cov = tf.einsum("ii",cov)
        ev_cov = tf.linalg.eigvalsh(cov)[-1]
        return [float(ev_cov),float(trace_cov)]
        

class training_objects_3D():
    def __init__(self,u_model,quad_rule):
        self.u_model=u_model
        self.quad_rule=quad_rule
    
        nval = 7
        self.h1_exact = 2.652285012
        
        self.fine_quad = Quadrature(3,"G3",nval)
            
        
        
    def u_exact(self,xyz):
        x,y,z=tf.unstack(xyz,axis=-1)
        return 10*tf.math.sin(np.pi*x)*tf.math.sin(np.pi*y)*tf.math.sin(np.pi*z)*((x - 2**(-1))**2 + (y - 2**(-1))**2 + (z - 2**(-1))**2 - 16**(-1))

    def rhs(self,xyz):
        x,y,z=tf.unstack(xyz,axis=-1)
        return ((((480 + (-240*x**2 - 240*y**2 - 240*z**2 + 240*x + 240*y + 240*z - 165)*np.pi**2)*tf.math.sin(np.pi*z) + 320*(z - 1/2)*tf.math.cos(np.pi*z)*np.pi)*tf.math.sin(np.pi*y) + 320*tf.math.cos(np.pi*y)*tf.math.sin(np.pi*z)*(y - 1/2)*np.pi)*tf.math.sin(np.pi*x))/8 + 40*tf.math.sin(np.pi*y)*(x - 1/2)*tf.math.cos(np.pi*x)*tf.math.sin(np.pi*z)*np.pi


    @tf.function
    def loss_function(self):
        x,y,z,w =self.quad_rule()
        
        
        with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as tx:
            with tf.autodiff.ForwardAccumulator(y, tf.ones_like(y)) as ty:
                with tf.autodiff.ForwardAccumulator(z, tf.ones_like(x)) as tz:
                    xyz = tf.stack([x,y,z],axis=-1)
                    v = tf.squeeze(self.u_model(xyz))
                dvz = tz.jvp(v)
                del tz
            dvy = ty.jvp(v)
            del ty
        dvx = tx.jvp(v)
        del tx
        
        
        RHS = self.rhs(xyz)
        
        integrand =   dvx**2/2+dvy**2/2+dvz**2/2 +v*RHS
        return tf.reduce_sum(integrand*w)
    
    
    @tf.function
    def loss_at_exact(self):
        x,y,z,w =self.quad_rule()
        
        
        with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as tx:
            with tf.autodiff.ForwardAccumulator(y, tf.ones_like(y)) as ty:
                with tf.autodiff.ForwardAccumulator(z, tf.ones_like(x)) as tz:
                    xyz = tf.stack([x,y,z],axis=-1)
                    v = tf.squeeze(self.u_exact(xyz))
                dvz = tz.jvp(v)
                del tz
            dvy = ty.jvp(v)
            del ty
        dvx = tx.jvp(v)
        del tx
        
        
        RHS = self.rhs(xyz)
        
        integrand =   dvx**2/2+dvy**2/2+dvz**2/2 +v*RHS
        return tf.reduce_sum(integrand*w)
    
    @tf.function
    def dloss_fn(self):
        with tf.GradientTape() as t1:
            loss = self.loss_function()
        dloss = t1.gradient(loss,self.u_model.trainable_weights)
        return loss,dloss 
    
    @tf.function
    def h1_error(self):
        x,y,z,w = self.fine_quad()
        with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as tx:
            with tf.autodiff.ForwardAccumulator(y, tf.ones_like(y)) as ty:
                with tf.autodiff.ForwardAccumulator(z, tf.ones_like(z)) as tz:
                    xyz = tf.stack([x,y,z],axis=-1)
                    ue = tf.squeeze(self.u_exact(xyz)-self.u_model(xyz))
                due_z = tz.jvp(ue)
            due_y = ty.jvp(ue)
        due_x = tx.jvp(ue)
        
        h1 = tf.reduce_sum(((due_x)**2+(due_y)**2+(due_z)**2)*w)**0.5
        
        return h1/self.h1_exact
    
    
    def estimate_var_loss(self,counts):
        tf.random.set_seed(1234)
        losslist = [self.loss_function() for i in range(counts)]
        return np.var(losslist)
    
    def estimate_var_exact(self,counts):
        tf.random.set_seed(1234)
        losslist = [self.loss_at_exact() for i in range(counts)]
        return np.var(losslist)
    
    def estimate_var_grad(self,counts):
        loss,dloss1 = self.dloss_fn()
        Ex = tf.zeros_like(tf.concat([tf.reshape(g,[-1]) for g in dloss1],-1))
        Ex2 = tf.einsum("i,j->ij",Ex,Ex)
        
        for i in range(counts):
            loss,dloss = self.dloss_fn()
            dloss = tf.concat([tf.reshape(g,[-1]) for g in dloss],-1)
            Ex2 += tf.einsum("i,j->ij",dloss,dloss)/counts
            Ex += dloss/counts
        cov = Ex2-tf.einsum("i,j->ij",Ex,Ex)
        trace_cov = tf.einsum("ii",cov)
        ev_cov = tf.linalg.eigvalsh(cov)[-1]
        return [float(ev_cov),float(trace_cov)]
        
    
    
    
    
        
