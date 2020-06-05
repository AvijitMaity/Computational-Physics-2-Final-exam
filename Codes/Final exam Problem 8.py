'''
====================================================================
Author: Avijit Maity
====================================================================
This Program solves the boundary value problem using

'''

import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt


a=0         #starting value
b=1       #end value

#We rewrite the equation as a first order system and
# #implement its right-hand side evaluation

def fun(x,y):
    return np.vstack((y[1],4*(y[0]-x)))

#Implement evaluation of the boundary condition residuals:
def bc(ya,yb):               #function for boundary conditions
    return np.array([ya[0],yb[0]-2])

x=np.linspace(a,b,100)       #creating mesh points
X=np.linspace(a,b,100)
y=np.zeros((2,x.size))
y[0]=0           #initial value

sol=solve_bvp(fun,bc,x,y)     #solving the problem using solve_bvp

# Defining analytical solution
def analytical(x):
   return (np.exp(1)**2 / (np.exp(1)**4 - 1)) * (np.exp(2*x) - np.exp(-2*x))+ x

plt.plot(x,sol.sol(x)[0],color='yellow',label="solve_bvp")    #plotting the solution using solve_bvp
plt.plot(x,analytical(x),'r--',label="Analytical")
plt.legend(loc=4)
plt.xlabel("x",size=13)
plt.ylabel("y",size=13)
plt.title( "Solving boundary value problem using scipy.integrate.solve_bvp" ,size=15)
plt.suptitle("Problem 8")
plt.grid()
plt.show()