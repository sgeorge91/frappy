# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 00:26:27 2022

@author: SVG
"""


import d2
from scipy.integrate import odeint
import numpy as np
from scipy.optimize import curve_fit
initial_state = [0.1, 0, 0]
sigma = 10.
rho = 28.
beta = 8./3.

start_time = 0
end_time = 200
time_points = np.linspace(start_time, end_time, end_time*100)
def lorenz_system(current_state, t):
    x, y, z = current_state

    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    
    return [dx_dt, dy_dt, dz_dt]
xyz = odeint(lorenz_system, initial_state, time_points)
x = xyz[:, 0]
op=d2.find_d2(x)
print(op)
xs=np.arange(len(op))
xs=xs+1
ys=[k[0] for k in op]
errs=[k[1] for k in op]
op2=curve_fit(d2.d2model, xs, ys,p0=[2,2] ,sigma=errs)
print(op2[0][1], "is D2\n", int(op2[0][0])+1, "is the embedding dimension")
