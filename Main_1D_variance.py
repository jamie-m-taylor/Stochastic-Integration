# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 11:06:22 2025

@author: jamie.taylor
"""


from Quadrature.Quadratures import Quadrature
from SRC.Losses import training_objects_1D
from SRC.Exports import export_variances

import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

quad_rules = ["MC","P0","P1","P3"]
order_list = [-1,0,1,3]
exponents = [-1-(2+2*p) for p in order_list]

pts_list = [10*2**i for i in range(11)]

all_pts_vars = []

tf.random.set_seed(1234)

for q in quad_rules:
    pts_vars = []
    for pts in pts_list:
        if q == "MC":
            N = pts
        elif q == "P0":
            N = int(pts)
        elif q == "P1":
            N = int(pts/2)
        elif q == "P3":
            N=int(pts/3)
            
        quad_rule = Quadrature(1,q,N)
        
        trainer = training_objects_1D(None,quad_rule)
        x,w = quad_rule()
        real_pts = len(x)
        pts_vars+= [ [real_pts,trainer.estimate_var_exact(1000)]]
    pts_vars = tf.constant(pts_vars)
    all_pts_vars +=[pts_vars]


coefficients = [all_pts_vars[i][-1,1]*all_pts_vars[i][-1,0]**(-exponents[i]) for i in range(4)]

for i in range(len(quad_rules)):
    plt.scatter(all_pts_vars[i][:,0],all_pts_vars[i][:,1],label=quad_rules[i])
    plt.plot(all_pts_vars[i][:,0],all_pts_vars[i][:,0]**(exponents[i])*coefficients[i],"--")

plt.xscale("log")
plt.yscale('log')
plt.show()

all_pts_vars = tf.concat(all_pts_vars,axis=-1)



print("Rules/Coefficients/Exponents",[[quad_rules[i],float(coefficients[i]),exponents[i]] for i in range(4)])

path = os.path.join("Data","Variance_1D")
name="Var_data"

export_variances(path, name, all_pts_vars)