'''
 Runge-Kutta Method
====================================================================
Author: Avijit Maity
====================================================================
This Program solves the initial value problem using
 with Runge-Kutta Method
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# First we will solve this problem using 4th order runge kutta method
# The ordinary differential equation
def f1(x,y1,y2 ):
    return 32*y1 + 66*y2 + (2/3)*x + (2/3)

def f2(x,y1,y2 ):
    return -66*y1 - 133*y2 - (1/3)*x - (1/3)

def f(x): # defining the function for justification
    return np.exp(-x) + x

# Define the initial values and mesh
# limits: 0.0 <= x <= 0.5
a=0
b=0.5
h= 0.001 # determine step-size
x = np.arange( a, b+h, h ) # create mesh
N = int((b-a)/h)
y1 = np.zeros((N+1,)) # initialize y1
y2 = np.zeros((N+1,)) # initialize y2

x[0]=0
y1[0]=1/3 # Initial value of y1
y2[0]=1/3 # Initial value of y2



for i in range(1,N+1): # Apply Runge-Kutta Method
    k1 = h * f1(x[i - 1], y1[i - 1], y2[i - 1])
    l1 = h * f2(x[i - 1], y1[i - 1], y2[i - 1])


    k2 = h * f1(x[i - 1] + h / 2.0, y1[i - 1] + k1 / 2.0, y2[i - 1] + l1 / 2.0 )
    l2 = h * f2(x[i - 1] + h / 2.0, y1[i - 1] + k1 / 2.0, y2[i - 1] + l1 / 2.0 )

    k3 = h * f1(x[i - 1] + h / 2.0, y1[i - 1] + k2 / 2.0, y2[i - 1] + l2 / 2.0)
    l3 = h * f2(x[i - 1] + h / 2.0, y1[i - 1] + k2 / 2.0, y2[i - 1] + l2 / 2.0)

    k4 = h * f1(x[i-1]+h, y1[i - 1] + k3, y2[i - 1] + l3)
    l4 = h * f2(x[i-1]+h, y1[i - 1] + k3, y2[i - 1] + l3)


    y1[i] = y1[i - 1] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    y2[i] = y2[i - 1] + (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0

# Now we will solve this problem using odient library function
# function that returns dz/dt
def model(z,x):
    dy1dt = 32*z[0] + 66*z[1] + (2/3)*x + (2/3)
    dy2dt = -66*z[0] - 133*z[1] - (1/3)*x - (1/3)
    dzdt = [dy1dt,dy2dt]
    return dzdt

# initial condition
z0 = [1/3,1/3]

# solve ODE
z = odeint(model,z0,x)



# Plot the RK4 solution
plt.subplot(1,2,1)
plt.plot(x, y1, label='y1 vs t using RK4' )
plt.plot( x, y2, label='y2 vs t using RK4' )
plt.title( "Solving the initial value problem using RK4 and Odient" )
plt.suptitle("Problem 6")
plt.xlabel('x')
plt.ylabel('y(x)')
# plot the solution using odient library function
plt.plot(x,z[:,0],'b--',label='y1 vs t using odient')
plt.plot(x,z[:,1],'r--',label='y2 vs t using odient')
plt.legend(loc='best')
plt.grid()

# here we will plot to justify our solution
plt.subplot(1,2,2)
plt.plot(x, 2*y1 + y2, label='2*y1 + y2',color="yellow" )
plt.plot(x, f(x), 'b--',label='exp(-x) + x' )
plt.xlabel('x')
plt.ylabel('2*y1+y2, f(x)')
plt.legend(loc='best')
plt.grid()

plt.show()
