import warnings
import multiprocessing as mp
import time
import numpy as np
warnings.filterwarnings("ignore")
from BP3MG.BP3MG_Synchronous.PAR3MG_MPI import PAR3MG_MPI
from scipy.io import loadmat
from BP3MG.PSF_tools.gaussian_kernel_3D import gaussian_kernel_3D
from BP3MG.blur_alt_z import blur_alt_z
from BP3MG.adjblur_alt_z import adjblur_alt_z
import matplotlib.pyplot as plt

#Nx = 16
#Ny = Nx
#Nz = 48
#I = loadmat('/home/mathieuchalvidal/PycharmProjects/untitled/Images/FlyBrain.mat')['I']
#sli = slice(0,256,int(256/16))
#I = I[sli,sli,:Nz]


Nx = 16
Ny = 16
Nz = 154
I = loadmat('/home/mathieuchalvidal/Downloads/bp3mg/Images/aneurysm2.mat')['I']
#sli = slice(0,155,int(155/16))
I = I[30:30+Nx,30:30+Nx,:Nz]

print('Create blurry and noisy image')
print('size image: Nx = ', Nx, ', Ny = ', Ny, ', Nz = ',Nz)

#Degradation parameters
Nh = np.array([5,5,11]).astype(int)
#Sx = loadmat('/home/mathieuchalvidal/Downloads/bp3mg/Sx.mat')['Sx']
#Sy = loadmat('/home/mathieuchalvidal/Downloads/bp3mg/Sy.mat')['Sy']
#Sz = loadmat('/home/mathieuchalvidal/Downloads/bp3mg/Sz.mat')['Sz']
Sx = np.random.rand(Nz, 1) * 3
Sy = np.random.rand(Nz, 1) * 3
Sz = np.random.rand(Nz, 1) * 4
Phiy = np.random.rand(Nz, 1) * 2 * np.pi * 0
#Phiz = loadmat('/home/mathieuchalvidal/Downloads/bp3mg/Phiz.mat')['Phiz']
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

#Iblurnoisy2 = loadmat('/home/mathieuchalvidal/Downloads/bp3mg/Iblurnoisy.mat',mat_dtype=True)['Iblurnoisy']
#Iblur2 = loadmat('/home/mathieuchalvidal/Downloads/bp3mg/Iblur.mat')['Iblur']

#Iblurnoisy3 = np.empty((64,64,24),np.longdouble)
#Iblurnoisy3[:,:,:]=Iblurnoisy2

#Iblur3 = np.empty((64,64,24),np.longdouble)
#Iblur3[:,:,:]=Iblur2

#print('max ecart', np.max(Iblur - Iblur2))

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

#Hty2 = np.swapaxes(loadmat('/home/mathieuchalvidal/Downloads/bp3mg/Hty.mat')['Hty'].reshape(64,64,Nz),0,1)
#H12 = loadmat('/home/mathieuchalvidal/Downloads/bp3mg/H1.mat',mat_dtype=True)['H1'].reshape(64,64,Nz)

#Hty3 = np.empty((64,64,24),np.longdouble)
#Hty3[:,:,:]=Hty2

#H13 = np.empty((64,64,Nz),np.longdouble)
#H13[:,:,:]=H12

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

#Hty_ = loadmat('/Users/GrandesTerres1/PycharmProjects/untitled/Seeds/Hty.mat')['Hty'].reshape(64,64,24)
#Hty_ = np.swapaxes(Hty_,0,1)

Times = {}
Timesending ={}
Crits = {}
ratio = {}
SNR = {}


for cores in [2,3,4,5,6,7,8,9,10,11,12]:
    BMMD = PAR3MG_MPI(y, h, Hty, H1, eta, tau, lambda_, delta, xmin, xmax, phi, x0, I, Nx, Ny, Nz, NbIt, cores, Timemax)
    BMMD.optimize()
    Crits[cores] = BMMD.Crit
    Times[cores] = np.cumsum(BMMD.Time)
    SNR[cores] = BMMD.SNR
    plt.plot(Times[cores],Crits[cores],label='BMMD_{}'.format(cores))

plt.yscale('log')
plt.legend()
plt.savefig('comp_local_BMMD')

for cores in [3,4,5,6,7,8,9,10,11,12]:
    ratio[cores] = (Times[2][-1]/Times[cores][-1])


plt.clf()
y_acc = np.poly1d(np.polyfit(list(ratio.keys()),list(ratio.values()),1))(list(ratio.keys()))
x_acc = list(ratio.keys())
plt.scatter(list(ratio.keys()),list(ratio.values()))
plt.plot(x_acc,y_acc)
plt.savefig('acceleration')

f = open("Times.txt","w")
f.write( str(Times) )
f.close()

f = open("Crits.txt","w")
f.write( str(Crits) )
f.close()

f = open("ratio.txt","w")
f.write( str(ratio) )
f.close()

f = open("SNR.txt","w")
f.write( str(SNR) )
f.close()

for cores in [5,7,9,12]:
    ratio[cores] = (len(Times[2])/len(Times[cores]))

plt.clf()
y_acc = np.poly1d(np.polyfit(list(ratio.keys()),list(ratio.values()),1))(list(ratio.keys()))
x_acc = list(ratio.keys())
plt.scatter(list(ratio.keys()),list(ratio.values()),marker='+',c='red')
plt.plot(x_acc,y_acc)
plt.title('Acceleration ratio in reconstruction of Flybrain with $\epsilon < 10^{-3}$ from 10 to 49 workers',fontsize=8)
plt.xlim([5,55]);plt.ylim([0,4])
plt.savefig('acceleration_ratio')
