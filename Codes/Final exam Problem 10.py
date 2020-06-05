from numpy import *
import  matplotlib.pyplot as plt

def f(x): # defining given function
    out = zeros(len(x), dtype=float64)
    for i in range(len(x)):
        if (-1.0 < x[i] and x[i] < 1.0):
            out[i] = 1.0
    return out

X = linspace(-50, 50, 1024) # x values for plotting box function


def fourier(n): # this function will compute fourier transformation for different n values
	xmin = -50
	xmax = 50
	dx = (xmax - xmin) / (n - 1)
	x = linspace(xmin, xmax, n)
	nft = fft.fft(f(x), norm='ortho')  # computing DFT
	k = fft.fftfreq(n, d=dx)  # Computing frequencies
	k = 2 * pi * k
	factor = exp(-1j * k * xmin)
	aft = dx * sqrt(n / (2.0 * pi)) * factor * nft
	# Now we will reorder arrays for plotting using fftshift
	k = fft.fftshift(k)
	aft = fft.fftshift(aft)
	return [k,aft]

fourier1=fourier(512)
fourier2=fourier(1024)
fourier3=fourier(2048)

plt.suptitle('Fourier tramsform of box function', size = 18, color = 'r')
# Here we will plot our original function
plt.subplot(2, 2, 1)
plt.plot(X,f(X), label ='Box function', color="red")
plt.xlim(-4,4)
plt.xlabel('x',size = 13)
plt.ylabel('f(x)',size = 13)
plt.legend(loc=4)
plt.tight_layout()
plt.grid()
plt.title("Configaration space",size = 14)

#Here we will plot Fourier transform of the box function for n = 512
plt.subplot(2, 2, 2)
plt.plot(fourier1[0],fourier1[1].real,label='Numerical', color="green")
plt.xlabel('frequency(k)',size = 13)
plt.ylabel("FT(box function)",size = 13)
plt.legend(loc=4)
plt.grid()
plt.tight_layout()
plt.title("Fourier transform for n = 512",size = 14)

#Here we will plot Fourier transform of the box function for n = 1024
plt.subplot(2, 2, 3)
plt.plot(fourier2[0],fourier2[1].real,label='Numerical', color="orange")
plt.xlabel('frequency(k)',size = 13)
plt.ylabel("FT(box function)",size = 13)
plt.legend(loc=4)
plt.grid()
plt.tight_layout()
plt.title("Fourier transform for n = 1024",size = 14)

#Here we will plot Fourier transform of the box function for n = 2048
plt.subplot(2, 2, 4)
plt.plot(fourier3[0],fourier3[1].real,label='Numerical')
plt.xlabel('frequency(k)',size = 13)
plt.ylabel("FT(box function)",size = 13)
plt.legend(loc=4)
plt.grid()
plt.tight_layout()
plt.title("Fourier transform for n = 2048",size = 14)




plt.show()
