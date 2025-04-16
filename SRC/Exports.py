# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 20:42:00 2025

@author: jamie.taylor
"""


import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

dtype="float64"

def export_u_1D(u_models, u_exact,npts,name,path,mid=False):
    
    if mid:
        x= tf.constant([((i+0.5)/npts) for i in range(0,npts)],dtype=dtype)
    else:
        x = tf.constant([((i)/npts) for i in range(0,npts+1)],dtype=dtype)
        
     
    with tf.GradientTape(persistent=True) as t1:
        t1.watch(x)
        u_ex = u_exact(x)
    dux_ex = t1.gradient(u_ex,x)
    del t1
    
    data = [x,u_ex,dux_ex]
    
    for u in u_models:
        
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            u_ap=u(x)
        dux_ap = t1.gradient(u_ap,x)
        
        del t1
        data+=[u_ap,dux_ap,u_ap-u_ex,dux_ap-dux_ex]
    
    
    data = tf.stack(data,axis=-1)
    
    np.savetxt(os.path.join(path,name)+".csv",data,delimiter=",",fmt='%1.16f')
    
    
def log_bin_data(x, y, N, start):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Define logarithmically spaced bin edges
    bins = np.exp(np.linspace(np.log(start), np.log(x[-1]), N + 1))
    
    # Compute bin indices for each x value
    bin_indices = np.searchsorted(bins, x, side='right') - 1
    
    # Mask out-of-range values
    valid_mask = (bin_indices >= 0) & (bin_indices < N)
    bin_indices = bin_indices[valid_mask]
    y_valid = y[valid_mask]
    x_valid = x[valid_mask]
    
    # Compute mean y values for each bin
    sums = np.bincount(bin_indices, weights=y_valid, minlength=N)
    counts = np.bincount(bin_indices, minlength=N)
    avg_y = np.divide(sums, counts, where=counts > 0, out=np.zeros_like(sums))
    
    # Compute geometric mean for each bin
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    
    return bin_centers, avg_y


def export_log_bin(path,name,training_objects,N,start,mode="loss"):
    all_data = []
    for t in training_objects:
        px = t.epochs
        if mode=="loss":
            py = t.loss_list
        elif mode == "h1":
            py = t.h1_list
        x,y = log_bin_data(px,py,N,start)
        all_data+=[x,y]
    all_data = tf.einsum("ij->ji",tf.constant(all_data))
    np.savetxt(os.path.join(path,name)+".csv",all_data,delimiter=",",fmt='%1.16f')
    
def compute_statistics_box(data):
    # Convert data into a NumPy array for easier manipulation
    data = np.array(data)
    
    # Calculate the required statistics along the specified axis (axis=1 for each row)
    min_vals = np.min(data, axis=1)
    lower_quartile = np.percentile(data, 25, axis=1)
    median_vals = np.median(data, axis=1)
    upper_quartile = np.percentile(data, 75, axis=1)
    max_vals = np.max(data, axis=1)
    
    # Stack them into a 5xN array
    result = np.vstack([min_vals, lower_quartile, median_vals, upper_quartile, max_vals])
    
    return result

def export_box(path,name,training_objects,mode="loss"):
    if mode=="loss":
        data = [t.loss_list[-1000:] for t in training_objects]
    elif mode=="h1":
        data = [t.h1_list[-1000:] for t in training_objects]
    result = compute_statistics_box(data).T
    np.savetxt(os.path.join(path,name)+".csv",result,delimiter=",",fmt='%1.16f')
    
def export_variances(path,name,varlists):
    np.savetxt(os.path.join(path,name)+".csv",varlists,delimiter=",",fmt='%1.32f')
    