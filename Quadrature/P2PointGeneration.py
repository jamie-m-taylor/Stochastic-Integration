# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:37:48 2025

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np
import os
import csv
dtype="float64"


def in_triangle(x,y):
    if y<-0.5:
        ans = False
    else:
        if y>(3**0.5)*x+1:
            ans = False
        else:
            if y>-(3**0.5)*x+1:
                ans=False
            else:
                ans=True
    return ans

def gen_point_P2_tri():
    found = False
    while not found:
        r =  np.random.random()**(1/4)
        t= np.random.random()*2*np.pi
        x = r*np.cos(t)
        y = r*np.sin(t)
        found = in_triangle(x,y)
    return x,y

def gen_points_P2_tri(n):
    xlist = []
    ylist= []
    for i in range(n):
        x,y = gen_point_P2_tri()
        xlist+=[x]
        ylist+=[y]
    return tf.einsum("ij->ji",tf.constant([xlist,ylist],dtype=dtype))


def save_points_tri(path,name,n,seed=1234):
    tf.random.set_seed(seed)
    xy = gen_points_P2_tri(n)
    np.savetxt(os.path.join(path,name)+".csv",xy,delimiter=",",fmt='%1.16f')  


###Tests if a 3-vector is in the tetrahedron
def in_tet(x):
    v1=tf.constant([4,-4,-4],dtype=dtype)
    ans = False
    if tf.reduce_sum(x*v1)>-4:
        v2 = tf.constant([4,4,-4],dtype=dtype)
        if tf.reduce_sum(x*v2)<4:
            v3 = tf.constant([-4,4,-4],dtype=dtype)
            if tf.reduce_sum(x*v3)>-4:
                v4 = tf.constant([-4,-4,-4],dtype=dtype)
                if tf.reduce_sum(x*v4)<4:
                    ans = True
    return ans


def make_tet_samp_P2_tet(n):
    tetlist = []
    i=0
    while len(tetlist)<n:
        tf.print(len(tetlist))
        i+=1
        r =  (tf.random.uniform([1],dtype=dtype)**(1/5))*3**0.5
        px = tf.random.normal([3],stddev=2,dtype=dtype)
        x = r*px/tf.reduce_sum(px**2)**0.5
        if in_tet(x):
            tetlist+=[x]
    return tf.stack(tetlist,axis=0)

def save_points_tet(path,name,n,seed=1234):
    xyz = make_tet_samp_P2_tet(n)
    np.savetxt(os.path.join(path,name)+".csv",xyz,delimiter=",",fmt='%1.16f')  
