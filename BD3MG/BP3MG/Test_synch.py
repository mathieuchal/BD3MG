import warnings
import multiprocessing as mp
import time
import numpy as np
from scipy.io import loadmat
warnings.filterwarnings("ignore")

from BP3MG.PAR3MG_MPI import PAR3MG_MPI
from PSF_tools.gaussian_kernel_3D import gaussian_kernel_3D
from BD3MG.blur_alt_z import blur_alt_z
from BD3MG.adjblur_alt_z import adjblur_alt_z


#Loading image
I = loadmat(os.getcwd() + '/Images/FlyBrain.mat')['I']
Nx = 128
Ny = Nx
Nz = 24
sli = slice(0,256,int(256/Nx))
I = I[sli,sli,:Nz]

print('Create blurry and noisy image')
print('size image: Nx = ', Nx, ', Ny = ', Ny, ', Nz = ',Nz)

#Degradation parameters
Nh = np.array([5,5,11]).astype(int)
Sx = np.random.rand(Nz, 1) * 3
Sy = np.random.rand(Nz, 1) * 3
Sz = np.random.rand(Nz, 1) * 4
Phiy = np.random.rand(Nz, 1) * 2 * np.pi * 0
Phiz = np.random.rand(Nz, 1) * 2 * np.pi


#getting convolution kernel
def h(z):
    return gaussian_kernel_3D(((Nh-1)/2).astype(int), [Sx[z], Sy[z], Sz[z]], [Phiy[z], Phiz[z]])
print('size kernel: Nx = {}, Ny = {}, Nz = {}'.format(*h(0).shape))

#SNR = 20;
sigma = 0.02

#add blur and noise in a parrallel fashion
start = time.time()
pool = mp.Pool(mp.cpu_count())
results = [pool.apply(blur_alt_z, args=(I, Nh, Nx, Ny, Sx, Sy, Sz, Phiy, Phiz, sigma, z)) for z in range(Nz)]
pool.close()

Iblurnoisy, Iblurz, BSNRinitz, SNRinitz  = zip(*results)
Iblurnoisy = np.dstack(Iblurnoisy)
Iblur = np.dstack(Iblurz)


cpu_time_blur = time.time()-start

SNRinit = 10*np.log10(np.sum(I**2)/np.sum((I-Iblurnoisy)**2))
BSNRinit = 10*np.log10(np.sum(I**2)/np.sum((I-Iblur)**2))
print('SNR init = ', str(SNRinit),', BSNRinit = ', str(BSNRinit))


y = Iblurnoisy.reshape(Nx*Ny, Nz)
#we need these vectors in the algorithm

pool = mp.Pool(mp.cpu_count())
results = [pool.apply(adjblur_alt_z, args=(Iblurnoisy, z, Nh, Nx, Ny, Nz, Sx, Sy, Sz, Phiy, Phiz)) for z in range(Nz)]
pool.close()

Hty, H1Z  = zip(*results)
Hty= np.dstack(Hty)
H1 = np.dstack([H1Z[z][:,:,z] for z in range(len(H1Z))])


print('Elapsed time : ', cpu_time_blur)
print('done')


Timemax = 600
NbIt = 10000 #Max iterations number
#Regularization parameters:
lambda_ = 1
delta = 2
phi = 4
#Bounds of the constrained domain:
xmin = 0
xmax = 1
#Elastic net parameter:
tau = 1e-3
#Weight of the quadratic distance function :Gradz
eta = 0.1
#Initialization
x0 = np.zeros((Nx, Ny, Nz))


Times = {}
Timesending ={}
Crits = {}
ratio = {}
SNR = {}
cores=mp.cpu_count()


BMMD = PAR3MG_MPI(y, h, Hty, H1, eta, tau, lambda_, delta, xmin, xmax, phi, x0, I, Nx, Ny, Nz, NbIt, cores, Timemax)
BMMD.optimize()
Crits[cores] = BMMD.Crit
Times[cores] = np.cumsum(BMMD.Time)
SNR[cores] = BMMD.SNR


f = open("Times.txt","w")
f.write( str(Times) )
f.close()

f = open("Crits.txt","w")
f.write( str(Crits) )
f.close()

f = open("SNR.txt","w")
f.write( str(SNR) )
f.close()

