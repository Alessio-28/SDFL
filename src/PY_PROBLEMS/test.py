# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 17:02:10 2020

@author: tomma
"""
import probs
import numpy as np
        
prb = probs.prob_collection
keys= list(prb.keys())

if __name__ == '__main__':
        
    #set constrained
    constrained = True

    #set number of problems
    nprob = 0
    print('problem         :  n nc ni | ma mb mc md me mf mz |       finit   viola   violb   violc   viold   viole   violf')
    print('---------------------------+----------------------+-------------------------------------------------------------')
    #      colville1       :  5  3  2 |  3  3  4  4  3  1  0 |                1.00    2.50    1.00    0.00    1.00    1.00
    for i,V in prb.items():
    #for i,V in [('prob10',prb['prob10'])]:
        if V.n >= 4:
            print('%-15s : %2d %2d %2d |' % (V.name,V.n,V.ncont,V.nint),end='')
            for j,m in V.m.items():
                if (constrained and m > 0) or (not constrained and m == 0):
                    nprob += 1
                print(' %2d' % m,end='')
            print(' |',end='')
            print(' %11.4e' % V.func_f(V.x_initial),end='')
            if V.n > 2:
                print(' %7.2f' % np.maximum(0.0,np.max(V.fconstr_a(V.x_initial))),end='')
                print(' %7.2f' % np.maximum(0.0,np.max(V.fconstr_b(V.x_initial))),end='')
            else:
                print('        ',end='')
                print('        ',end='')
            
            print(' %7.2f' % np.maximum(0.0,np.max(V.fconstr_c(V.x_initial))),end='')
            print(' %7.2f' % np.maximum(0.0,np.max(V.fconstr_d(V.x_initial))),end='')
            if V.n > 2:
                print(' %7.2f' % np.maximum(0.0,np.max(V.fconstr_e(V.x_initial))),end='')
            else:
                print('        ',end='')

            print(' %7.2f' % np.maximum(0.0,np.max(V.fconstr_f(V.x_initial))),end='')
            #print(V.fconstr_a(V.x_initial),end='')
            print()
                
    print(' Number of problems : ',nprob)