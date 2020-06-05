import numpy as np
import time
A = np.array([[1,1,0],[1,0,1],[0,1,1]])


# using numpy.linalg.svd.
U2,S2,V2=np.linalg.svd(A)# U2,V2,S2 are the decomposed  matrics through numpy.linalg.svd
print('SVD form using np.linalg.svd: \n')
print('U2:\n',U2,'\nS2:\n Singular values are for second matrix',S2,'\n V2:\n',V2)