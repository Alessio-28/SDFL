# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "maxquad"
startp    : npt.NDArray[np.float64] = np.zeros(10, dtype = np.float64)
lb        : npt.NDArray[np.float64] = startp - 10
ub        : npt.NDArray[np.float64] = startp + 10
n         : int = len(lb)
nint      : int = 5
ncont     : int = n - nint
lbmix     : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64);      lbmix[:ncont]     = lb[:ncont]
ubmix     : npt.NDArray[np.float64] = 100 * np.ones(n, dtype = np.float64); ubmix[:ncont]     = ub[:ncont]
x_initial : npt.NDArray[np.float64] =  50 * np.ones(n, dtype = np.float64); x_initial[:ncont] = (ub[:ncont] + lb[:ncont]) / 2 
xmix      : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    j = np.arange(1, 11, dtype = np.float64).reshape(-1,1)
    JK = np.triu(np.matmul(j, j.T), 1) + np.triu(np.matmul(j, j.T), 1).T
    JonK = np.triu(np.matmul(j, (j ** (-1)).T), 1) + np.triu(np.matmul(j, (j ** (-1)).T), 1).T
    A1 = np.exp(JonK) * np.cos(JK) * np.sin(1); A1 += np.diag(np.squeeze(j / 10 * np.sin(1) + np.sum(np.sum(np.abs(A1)))))
    A2 = np.exp(JonK) * np.cos(JK) * np.sin(2); A2 += np.diag(np.squeeze(j / 10 * np.sin(2) + np.sum(np.sum(np.abs(A2)))))
    A3 = np.exp(JonK) * np.cos(JK) * np.sin(3); A3 += np.diag(np.squeeze(j / 10 * np.sin(3) + np.sum(np.sum(np.abs(A3)))))
    A4 = np.exp(JonK) * np.cos(JK) * np.sin(4); A4 += np.diag(np.squeeze(j / 10 * np.sin(4) + np.sum(np.sum(np.abs(A4)))))
    A5 = np.exp(JonK) * np.cos(JK) * np.sin(5); A5 += np.diag(np.squeeze(j / 10 * np.sin(5) + np.sum(np.sum(np.abs(A5)))))
    
    b1 = np.exp(j / 1) * np.sin(j * 1);
    b2 = np.exp(j / 2) * np.sin(j * 2);
    b3 = np.exp(j / 3) * np.sin(j * 3);
    b4 = np.exp(j / 4) * np.sin(j * 4);
    b5 = np.exp(j / 5) * np.sin(j * 5);
    
    fx = np.zeros(5, dtype = np.float64)
    
    fx[0] = np.matmul(x.T, np.matmul(A1, x)) - np.matmul(x.T, b1)
    fx[1] = np.matmul(x.T, np.matmul(A2, x)) - np.matmul(x.T, b2)
    fx[2] = np.matmul(x.T, np.matmul(A3, x)) - np.matmul(x.T, b3)
    fx[3] = np.matmul(x.T, np.matmul(A4, x)) - np.matmul(x.T, b4)
    fx[4] = np.matmul(x.T, np.matmul(A5, x)) - np.matmul(x.T, b5)
    
    return np.max(fx)
