# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:44:41 2025

@author: jamie.taylor
"""

from Quadrature.QuadratureTriangles import Quad_triangles
from Quadrature.QuadratureTetrahedra import Quad_tetrahedra
from Quadrature.Quad1D import Quadrature_1D
from Quadrature.Quad2D import Quadrature_2D
from Quadrature.Quad3D import Quadrature_3D


class Quadrature():
    def __init__(self,dim,rule,N):
        if dim==1:
            self.Q = Quadrature_1D(rule,N)
        if dim==2:
            if rule =="P1t" or rule=="P2t" or rule=="P0t":
                self.Q = Quad_triangles(rule,N)
            else:
                self.Q = Quadrature_2D(rule,N)
        if dim==3:
            if rule =="P1t" or rule=="P2t":
                self.Q = Quad_tetrahedra(rule,N)
            else:
                self.Q = Quadrature_3D(rule,N)
    
    def __call__(self):
        return self.Q()