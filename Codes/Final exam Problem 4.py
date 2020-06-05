'''
Problem 4
====================================================================
Author: Avijit Maity
====================================================================
'''

import numpy as np
import matplotlib.pyplot as plt

# Part a(Plotting the sample)
N= 1024
random= np.random.rand(1024) #it will generate 1024 uniformly distributed random numbers between 0 and 1.
n=np.arange(1,N+1,1)

plt.subplot(2,2,1)
plt.scatter(n,random,color="green")
plt.xlabel('number count(n)',size=13)
plt.ylabel('random numbers',size=13)
plt.tight_layout()
plt.title("Random no. vs number count plot", size=15)


# Part b(Computing the power spectrum for this sample)

nft =np. fft.fft(random, norm='ortho')  # computing DFT
dx=1
k = np.fft.fftfreq(N, d=dx)  # Computing frequencies
k = 2 * np.pi * k
k = np.fft.fftshift(k) # Now we will reorder arrays for plotting using fftshift
nft=np.fft.fftshift(nft)

Power_spec=np.zeros(len(k))
for i in range(len(k)):
	Power_spec[i]=abs(nft[i])**2/len(k)

plt.subplot(2,2,2)
plt.plot(k,Power_spec, color="red")
plt.xlabel('frequency(k)',size=13)
plt.ylabel('spectral density',size=13)
plt.tight_layout()
plt.title("Power spectra using periodogram",size=15)


# Part c ( the minimum and maximum value of the wavevector k )

print("The minimum value of the wavevector k = ",min(k))
print("The maximum value of the wavevector k = ",max(k))

# Part d (Plotting the power spectrum in five uniform k bins)
k_span=int((max(k)-min(k)))
bin=5
width=k_span/bin


k1=np.linspace(min(k),max(k),bin+1)

Power_bin=np.zeros(bin)
k_bin=np.zeros(bin)


for i in range(bin):
	count=0
	for j in range(len(k)):
		if k1[i]<=k[j]<k1[i+1]:
			Power_bin[i]+=Power_spec[j]
			count+=1
	Power_bin[i]=Power_bin[i]/count
	k_bin[i]=k1[i]+(k1[i+1]-k1[i])/2

plt.subplot(2,2,3)
plt.bar(k_bin,Power_bin,width,color="orange")
plt.xlabel("Binned K")
plt.ylabel("Binned power spectra")
plt.tight_layout()
plt.title("Plot of binned power spectra for uniform distribution")

plt.show()