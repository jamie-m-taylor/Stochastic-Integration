# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 20:08:50 2025

@author: jamie.taylor
"""


import tensorflow as tf
import matplotlib.pyplot as plt
import os

from SRC.Architectures import make_u_model_3D
from SRC.Training import training
from Quadrature.Quadratures import Quadrature
from SRC.Losses import training_objects_3D
from SRC.Exports import export_log_bin, export_box


dtype="float64"

neurons = 30
iterations=15000




# N_P2_poor= 3
# N_P2_fine = [4,5,5]
# N_MC_poor = 13*5*3*3*3
# N_MC_high = 13*5*4*5*5

# ###I think this can be like halved
# N_MC_high_2 = 48750

N_P2_poor = 3
N_MC_poor = 13*5*3*3*3
N_MC_high =5320
N_MC_high_2 = 31000


u_model_MC_poor = make_u_model_3D(30)
u_model_MC_fine = make_u_model_3D(30)
u_model_MC_fine_2 = make_u_model_3D(30)
u_model_P2_poor = make_u_model_3D(30)
# u_model_P2_fine = make_u_model_3D(30)

quad_rule_P2_poor = Quadrature(3,"P2t",N_P2_poor)
# quad_rule_P2_fine = Quadrature(3,"P2t",N_P2_fine)
quad_rule_MC_poor = Quadrature(3,"MC",N_MC_poor)
quad_rule_MC_fine = Quadrature(3,"MC",N_MC_high)
quad_rule_MC_fine_2 = Quadrature(3,"MC",N_MC_high_2)

training_objects_P2_poor = training_objects_3D(u_model_P2_poor,quad_rule_P2_poor)
# training_objects_P2_fine = training_objects_3D(u_model_P2_fine,quad_rule_P2_fine)
training_objects_MC_poor = training_objects_3D(u_model_MC_poor,quad_rule_MC_poor)
training_objects_MC_fine = training_objects_3D(u_model_MC_fine,quad_rule_MC_fine)
training_objects_MC_fine_2 = training_objects_3D(u_model_MC_fine_2,quad_rule_MC_fine_2)

trainer_P2_poor = training(training_objects_P2_poor,iterations)
# trainer_P2_fine = training(training_objects_P2_fine,iterations)
trainer_MC_poor = training(training_objects_MC_poor,iterations)
trainer_MC_fine = training(training_objects_MC_fine,iterations)
trainer_MC_fine_2 = training(training_objects_MC_fine_2,iterations)


trainer_P2_poor.train()
# trainer_P2_fine.train()
trainer_MC_poor.train()
trainer_MC_fine.train()
trainer_MC_fine_2.train()


all_training_objects = [trainer_MC_poor, 
                        trainer_MC_fine, 
                        trainer_MC_fine_2,
                        trainer_P2_poor] 
                        # trainer_P2_fine]


plt.plot(tf.stack(trainer_P2_poor.h1_list),label="P2 poor")
# plt.plot(tf.stack(trainer_P2_fine.h1_list),label="P2 fine")
plt.plot(tf.stack(trainer_MC_poor.h1_list),label="MC poor")
plt.plot(tf.stack(trainer_MC_fine.h1_list),label="MC fine")
plt.plot(tf.stack(trainer_MC_fine_2.h1_list),label="MC fine 2")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()


var_lists = [training_objects_MC_poor.estimate_var_loss(500),
              training_objects_MC_fine.estimate_var_loss(500),
              training_objects_MC_fine_2.estimate_var_loss(500),
              training_objects_P2_poor.estimate_var_loss(500)]
              # training_objects_P2_fine.estimate_var_loss(500)]

var_lists_grads = [training_objects_MC_poor.estimate_var_grad(500),
              training_objects_MC_fine.estimate_var_grad(500),
              training_objects_MC_fine_2.estimate_var_grad(500),
              training_objects_P2_poor.estimate_var_grad(500)]
              # training_objects_P2_fine.estimate_var_grad(500)]


var_lists_exact = [training_objects_MC_poor.estimate_var_exact(500),
             training_objects_MC_fine.estimate_var_exact(500),
             training_objects_MC_fine_2.estimate_var_exact(500),
             training_objects_P2_poor.estimate_var_exact(500)]
             # training_objects_P2_fine.estimate_var_exact(500)]


path=os.path.join("Data","DRM_3D")

export_log_bin(path, "loss", all_training_objects, 25, 100,mode="loss")

export_log_bin(path, "h1_er", all_training_objects, 25, 100,mode="h1")


export_box(path, "loss_box", all_training_objects,mode="loss")

export_box(path, "h1_box", all_training_objects,mode="h1")

