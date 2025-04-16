# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 08:43:44 2025

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
iterations = 4000



u_model = make_u_model_1D(neurons)


training_objects = training_objects_1D(u_model,Quadrature(1,"G1",20))

trainer = training(training_objects,iterations)

all_training_objects = [trainer]


trainer.train()


path=os.path.join("Data","DRM_1D_det")

export_log_bin(path, "loss", all_training_objects, 25, 10,mode="loss")

export_log_bin(path, "h1_er", all_training_objects, 25, 10,mode="h1")


export_box(path, "loss_box", all_training_objects,mode="loss")

export_box(path, "h1_box", all_training_objects,mode="h1")


export_u_1D([u_model], training_objects.u_exact, 500, "sols", path)
export_u_1D([u_model], training_objects.u_exact, 20, "sols_quad_points", path,mid=True)

