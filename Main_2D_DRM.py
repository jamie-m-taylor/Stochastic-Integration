# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:44:41 2025

@author: jamie.taylor
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import os


from SRC.Architectures import make_u_model_2D
from SRC.Training import training
from Quadrature.Quadratures import Quadrature
from SRC.Losses import training_objects_2D
from SRC.Exports import export_log_bin, export_box


dtype="float64"
tf.keras.backend.set_floatx(dtype)

neurons = 30
iterations = 20000

N_P2_poor=4
N_P2_fine = 17
N_MC_poor = 128
N_MC_fine = 2312

u_model_MC_poor = make_u_model_2D(neurons)
u_model_MC_fine = make_u_model_2D(neurons)
u_model_P2_poor = make_u_model_2D(neurons)
u_model_P2_fine = make_u_model_2D(neurons)


training_objects_P2_poor = training_objects_2D(u_model_P2_poor,Quadrature(2,"P2t",N_P2_poor))
training_objects_P2_fine = training_objects_2D(u_model_P2_fine,Quadrature(2,"P2t",N_P2_fine))
training_objects_MC_poor = training_objects_2D(u_model_MC_poor,Quadrature(2,"MC",N_MC_poor))
training_objects_MC_fine = training_objects_2D(u_model_MC_fine, Quadrature(2,"MC",N_MC_fine))

trainer_P2_poor = training(training_objects_P2_poor,iterations)
trainer_P2_fine = training(training_objects_P2_fine,iterations)
trainer_MC_poor = training(training_objects_MC_poor,iterations)
trainer_MC_fine = training(training_objects_MC_fine,iterations)

all_training_objects = [trainer_MC_poor, 
                        trainer_MC_fine, 
                        trainer_P2_poor, 
                        trainer_P2_fine]


trainer_P2_poor.train()
trainer_P2_fine.train()
trainer_MC_fine.train()
trainer_MC_poor.train()

plt.plot(tf.stack(trainer_P2_poor.h1_list),label="P2 poor")
plt.plot(tf.stack(trainer_P2_fine.h1_list),label="P2 fine")
plt.plot(tf.stack(trainer_MC_poor.h1_list),label="MC poor")
plt.plot(tf.stack(trainer_MC_fine.h1_list),label="MC fine")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()


var_lists = [training_objects_P2_poor.estimate_var_loss(1000),
             training_objects_P2_fine.estimate_var_loss(1000),
             training_objects_MC_poor.estimate_var_loss(1000),
             training_objects_MC_fine.estimate_var_loss(1000)]


var_lists_exact = [training_objects_P2_poor.estimate_var_exact(1000),
             training_objects_P2_fine.estimate_var_exact(1000),
             training_objects_MC_poor.estimate_var_exact(1000),
             training_objects_MC_fine.estimate_var_exact(1000)]

var_lists_grads = [training_objects_MC_poor.estimate_var_grad(500),
              training_objects_MC_fine.estimate_var_grad(500),
              training_objects_P2_poor.estimate_var_grad(500),
               training_objects_P2_fine.estimate_var_grad(500)]

path=os.path.join("Data","DRM_2D")

export_log_bin(path, "loss", all_training_objects, 25, 100,mode="loss")

export_log_bin(path, "h1_er", all_training_objects, 25, 100,mode="h1")


export_box(path, "loss_box", all_training_objects,mode="loss")

export_box(path, "h1_box", all_training_objects,mode="h1")
