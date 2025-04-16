# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 16:53:29 2025

@author: jamie.taylor
"""


import tensorflow as tf
import numpy as np

dtype="float64"

tf.keras.backend.set_floatx(dtype)


pi = tf.constant(np.pi,dtype=dtype)


class my_linear_layer_smooth_1D(tf.keras.layers.Layer):
    def __init__(self):
        super(my_linear_layer_smooth_1D,self).__init__()
        # self.c = tf.Variable(0.,dtype=dtype,trainable=True)
    def call(self,inputs):
        L,x = inputs
        cut = x*(1-x)/0.577
        # return tf.einsum("ij->i",L*cut)*self.c
        return tf.einsum("ij->i",L*cut)
    

class squeeze_layer(tf.keras.layers.Layer):
    def call(self,x):
        return tf.einsum("ij->i",x)       

def make_u_model_1D(neurons,activation=tf.math.tanh):
    
    
    tf.random.set_seed(1234)
    init = tf.keras.initializers.GlorotUniform(seed=1)
    bias_init=tf.keras.initializers.RandomUniform(seed=1,minval=-0.25,maxval=0.25)
    
    
    xvals = tf.keras.layers.Input(shape=(1,),dtype=dtype)
    
    
    l1 = tf.keras.layers.Dense(neurons,activation=activation,kernel_initializer=init,dtype=dtype,
                               bias_initializer = bias_init)(xvals-0.5)
    
    l1 = tf.keras.layers.Dense(neurons,activation=activation,kernel_initializer=init,dtype=dtype,
                               bias_initializer = bias_init)(l1)
    
    l1 = tf.keras.layers.Dense(neurons,activation=activation,kernel_initializer=init,dtype=dtype,
                               bias_initializer = bias_init)(l1)
    
    l2 = tf.keras.layers.Dense(1,kernel_initializer=init,dtype=dtype,
                               bias_initializer = bias_init)(l1)
    u_out = my_linear_layer_smooth_1D()([l2,xvals])
    
    u_model = tf.keras.Model(inputs=xvals,outputs = u_out)
    
    u_model.summary()
    
    return u_model


class my_linear_layer_smooth_2D(tf.keras.layers.Layer):
    def __init__(self):
        super(my_linear_layer_smooth_2D,self).__init__()
    def call(self,inputs):
        L = inputs[0]
        px = inputs[1]
        # x,y=tf.unstack(px,axis=-1)
        x = px[:,0]
        y = px[:,1]
        # cut = tf.math.sin(np.pi*x)*tf.math.sin(np.pi*y)
        cut = x*(1-x)*y*(1-y)/0.149
        
        
        
        return (tf.einsum("ij->i",L))*cut

class my_linear_layer_smooth_3D(tf.keras.layers.Layer):
    def __init__(self):
        super(my_linear_layer_smooth_3D,self).__init__()
    def call(self,inputs):
        L = inputs[0]
        px = inputs[1]
        # x,y=tf.unstack(px,axis=-1)
        x = px[:,0]
        y = px[:,1]
        z = px[:,2]
        # cut = tf.math.sin(np.pi*x)*tf.math.sin(np.pi*y)*tf.math.sin(np.pi*z)
        cut = x*(1-x)*y*(1-y)*z*(1-z)/0.03333
        # cut = x*(1-x)*y*(1-y)*z*(1-z)
        
        
        
        return (tf.einsum("ij->i",L))*cut

def make_u_model_2D(neurons,activation=tf.math.tanh):
    
    
    tf.random.set_seed(1234)
    init = tf.keras.initializers.GlorotUniform(seed=1)
    bias_init=tf.keras.initializers.RandomUniform(seed=1,minval=-0.25,maxval=0.25)
    xvals = tf.keras.layers.Input(shape=(2,),dtype=dtype)
    
    offset = tf.constant([1/2,1/2],dtype=dtype)
    l1 = tf.keras.layers.Dense(neurons,activation=activation,kernel_initializer=init,
                                bias_initializer=bias_init)(xvals-offset)
    
    
    l1 = tf.keras.layers.Dense(neurons,activation=activation,kernel_initializer=init,dtype=dtype,
                               bias_initializer = bias_init)(l1)
    
    l1 = tf.keras.layers.Dense(neurons,activation=activation,kernel_initializer=init,dtype=dtype,
                               bias_initializer = bias_init)(l1)
    
    l2 = tf.keras.layers.Dense(1,kernel_initializer=init,dtype=dtype,
                               bias_initializer = bias_init)(l1)
    
    u_out = my_linear_layer_smooth_2D()([l2,xvals])
    
    u_model = tf.keras.Model(inputs=xvals,outputs = u_out)
    
    u_model.summary()
    
    return u_model




def make_u_model_3D(neurons,activation=tf.math.tanh):
    
    tf.random.set_seed(1234)
    init = tf.keras.initializers.GlorotUniform(seed=1)
    binit=tf.keras.initializers.RandomUniform(seed=1,minval=-0.25,maxval=0.25)
    
    xvals = tf.keras.layers.Input(shape=(3,),dtype=dtype)
    
    offset = tf.constant([1/2,1/2,1/2],dtype=dtype)
    l1 = tf.keras.layers.Dense(neurons,activation=activation,kernel_initializer=init,
                               bias_initializer=binit)(xvals-offset)
    
    l1 = tf.keras.layers.Dense(neurons,activation=activation,kernel_initializer=init,
                               bias_initializer=binit)(l1)
    
    l1 = tf.keras.layers.Dense(neurons,activation=activation,kernel_initializer=init)(l1)    
    
    l2 = tf.keras.layers.Dense(1,kernel_initializer=init,
                               bias_initializer=binit)(l1)
    u_out = my_linear_layer_smooth_3D()([l2,xvals])
    
    u_model = tf.keras.Model(inputs=xvals,outputs = u_out)
    
    u_model.summary()
    
    return u_model