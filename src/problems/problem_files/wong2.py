# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name   : str = "wong2"
startp : npt.NDArray[np.float64] = np.array([2, 3, 5, 5, 1, 2, 7, 3, 6, 10], dtype = np.float64)
n      : int = startp.size

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    x = x.reshape(-1,1)
    F = np.zeros(9, dtype = np.float64)
    F[0] = x[0]**2 + x[1]**2 + x[0]*x[1] - 14*x[0]- 16*x[1] + (x[2]-10)**2 + 4*(x[3]-5)**2+(x[4]-3)**2 + 2*(x[5]-1)**2 + 5*x[6]**2 + 7*(x[7]-11)**2+2*(x[8]-10)**2 + (x[9]-7)**2 + 45
    F[1] = F[0] + 10*(3*(x[0]-2)**2 + 4*(x[1]-3)**2 + 2*x[2]**2 -7*x[3]-120)
    F[2] = F[0] + 10*(5*x[0]**2 + 8*x[1] + (x[2]-6)**2 - 2*x[3] -40)
    F[3] = F[0] + 10*(0.5*(x[0]-8)**2 + 2*(x[1]-4)**2 + 3*x[4]**2 -x[5]-30)
    F[4] = F[0] + 10*(x[0]**2 + 2*(x[1]-2)**2 - 2*x[0]*x[1] + 14*x[4] - 6*x[5])
    F[5] = F[0] + 10*(-3*x[0] + 6*x[1] + 12*(x[8]-8)**2 - 7*x[9])
    F[6] = F[0] + 10*(4*x[0]+5*x[1]-3*x[6] +9*x[7]-105)
    F[7] = F[0] + 10*(10*x[0]-8*x[1]-17*x[6]+2*x[7])
    F[8] = F[0] + 10*(-8*x[0]+2*x[1]+5*x[8] -2*x[9]-12)

    return np.max(F)
