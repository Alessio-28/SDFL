import numpy as np
import gill
import goffin
import l1hilb
import osborne2
import polak2
import polak3
import shelldual
import steiner2
import watson
import wong2
import wong3

x = np.ones(10)
print(gill.feval(x))

x = np.arange(1,51)/10
print(goffin.feval(x))
print(l1hilb.feval(x))

x = osborne2.startp
print(osborne2.feval(x))

x = polak2.startp;
print(polak2.feval(x))

x = polak3.startp;
print(polak3.feval(x))

x = shelldual.startp;
print(shelldual.feval(x))

x = steiner2.startp;
print(steiner2.feval(x))

x = watson.startp;
print(watson.feval(x))

x = wong2.startp;
print(wong2.feval(x))

x = wong3.startp;
print(wong3.feval(x))