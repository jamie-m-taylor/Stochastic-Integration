# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 08:44:20 2025

@author: jamie.taylor
"""

import tensorflow as tf
import os

from Quadrature.Quadratures import Quadrature
from SRC.Training import training
from SRC.Losses import training_objects_1D
from SRC.Architectures import make_u_model_1D
from SRC.Exports import export_u_1D, export_log_bin, export_box

neurons=30 

Experiment = "Good"

if Experiment == "Poor":
    N = 6
    iterations = 100000
elif Experiment == "Moderate":
    N = 32
    iterations = 25000
elif Experiment =="Good":
    N = 252
    iterations = 10000




u_model_MC = make_u_model_1D(neurons)
u_model_P0 = make_u_model_1D(neurons)
u_model_P1 = make_u_model_1D(neurons)
u_model_P3 = make_u_model_1D(neurons)


training_objects_MC = training_objects_1D(u_model_MC,Quadrature(1,"MC",N))
training_objects_P0 = training_objects_1D(u_model_P0,Quadrature(1,"P0",N))
training_objects_P1 = training_objects_1D(u_model_P1,Quadrature(1,"P1",int(N/2)))
training_objects_P3 = training_objects_1D(u_model_P3, Quadrature(1,"P3",int(N/3)))

trainer_MC = training(training_objects_MC,iterations)
trainer_P0 = training(training_objects_P0,iterations)
trainer_P1 = training(training_objects_P1,iterations)
trainer_P3 = training(training_objects_P3,iterations)

all_training_objects = [trainer_MC, 
                        trainer_P0, 
                        trainer_P1, 
                        trainer_P3]


trainer_MC.train()
trainer_P0.train()
trainer_P1.train()
trainer_P3.train()


path=os.path.join("Data","DRM_1D_"+Experiment)

export_log_bin(path, "loss", all_training_objects, 25, 100,mode="loss")

export_log_bin(path, "h1_er", all_training_objects, 25, 100,mode="h1")


export_box(path, "loss_box", all_training_objects,mode="loss")

export_box(path, "h1_box", all_training_objects,mode="h1")

export_u_1D([u_model_MC,u_model_P0,u_model_P1,u_model_P3], training_objects_MC.u_exact, 200, "sols", path)

