'''
This program shows how to solve linear matrix equation using numpy.linalg.solve.
'''
import numpy as np
from numpy import linalg
A = np.array([[4,1,2], [2,4,-1], [1,1,-3]])
b = np.array([9,-5,-9])
x = linalg.solve(A, b)
print('The solution is',x)

B=np.allclose(np.dot(A, x), b) # It will check the solution is correct or not
print(B)

'''
Note-
After solving this matrix equation in Gaussian elimination method we get solution vector x=(1.1,-1,2.9).
Here we get solution vector using linalg.solve x= [ 1. -1.  3.].
Both are almost same.
'''