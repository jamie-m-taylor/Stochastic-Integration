# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 08:43:34 2025

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
iterations = 50000



u_model_P1 = make_u_model_1D(neurons)
u_model_P1b = make_u_model_1D(neurons)


training_objects_P1 = training_objects_1D(u_model_P1,Quadrature(1,"P1",32))
training_objects_P1b = training_objects_1D(u_model_P1b,Quadrature(1,"P1b",32))

trainer_P1 = training(training_objects_P1,iterations)
trainer_P1b = training(training_objects_P1b,iterations)

all_training_objects = [trainer_P1, 
                        trainer_P1b]


trainer_P1.train()
trainer_P1b.train()


path=os.path.join("Data","DRM_1D_bias")

export_log_bin(path, "loss", all_training_objects, 25, 100,mode="loss")

export_log_bin(path, "h1_er", all_training_objects, 25, 100,mode="h1")


export_box(path, "loss_box", all_training_objects,mode="loss")

export_box(path, "h1_box", all_training_objects,mode="h1")


export_u_1D([u_model_P1,u_model_P1b], training_objects_P1.u_exact, 200, "sols", path)


