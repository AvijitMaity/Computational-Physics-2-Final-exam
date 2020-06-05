import numpy as np
import time
A = np.array([[2,1],[1,0],[0,1]])

# Here we write a Python code to determine the Singular Value Decomposition of an m × n matrix.
evalue1,evector1=np.linalg.eigh(np.dot(A,np.transpose(A)))# eigenvalues and eigen vectors of A*A^T
evalue2,evector2=np.linalg.eigh(np.dot(np.transpose(A),A))#eigenvalues and eigen vectors of A^T*A
eigen_1=evalue1.argsort()[::-1]#sorting of evalue1
evalue1=evalue1[eigen_1]
U1=evector1[:,eigen_1] #sorting of evector1 through evalue1 ranks
eigen_2=evalue2.argsort()[::-1]#sorting of evalue2
evalue2=evalue2[eigen_2]
V1=evector2[:,eigen_2]#sorting of evector1 through evalue2 ranks
S1=np.dot(np.dot(np.transpose(U1),A),V1)
print('SVD form using python code: \n')
print('U1:\n',U1,'\n S1:\n Singular values are for first matrix',S1,'\n V1:\n',V1) # U1,V1,S1 are the decomposed  matrics through python3 code


# using numpy.linalg.svd.
U2,S2,V2=np.linalg.svd(A)# U2,V2,S2 are the decomposed  matrics through numpy.linalg.svd
print('SVD form using np.linalg.svd: \n')
print('U2:\n',U2,'\nS2:\n Singular values are for first matrix',S2,'\n V2:\n',V2)