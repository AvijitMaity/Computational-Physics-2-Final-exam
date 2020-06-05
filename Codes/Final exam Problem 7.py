'''
Linear congruential random number generator in Python
====================================================================
Author: Avijit Maity
====================================================================
'''

import numpy as np
import matplotlib.pyplot as plt
import time
def LCG(a,c,m,x): # defining LCG
     n=100
     results = []

     for i in range(n):
         x = (a * x + c) % m
         results.append(x)

     return results

LCG1= LCG(8,8,10,8) # Here initial seed comes back again a=8, c=8, m=10, x0= 8
LCG2= LCG(4,0,16,4) # here initial seed does not come back again a=4, c=0, m=16, x0= 4

plt.suptitle("Linear congruential random number generator in Python")
# Here we will plot densirty histogram of LCG where the initial seed comes back again
plt.subplot(1,2,1)
plt.hist(LCG1,10,density = True,facecolor = 'green',ec='black',label = "Density Histogram")
plt.title("LCG where the initial seed comes back again",size = 14)
plt.xlabel("x")
plt.ylabel("PDF")
plt.grid()
plt.legend()

# Here we will plot densirty histogram of LCG where the initial seed does not come back again
plt.subplot(1,2,2)
plt.hist(LCG2,10,density = True,facecolor = 'red',ec='black',label = "Density Histogram")
plt.title("LCG where the initial seed does not come back again",size = 14)
plt.xlabel("x")
plt.ylabel("PDF")
plt.grid()
plt.legend()

plt.show()

