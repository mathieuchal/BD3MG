
import warnings
import os
from BP3MG.BP3MG_Synchronous.ComputeCriterionPar import ComputeCriterionPar
from BP3MG.BP3MG_Synchronous.PAR3MG_Master_Slave import PAR3MG_Master_worker
from BP3MG.BP3MG_Synchronous.set_blocks import set_blocks
import multiprocessing as mp
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
warnings.filterwarnings("ignore")


class PAR3MG_MPI:
    
    def __init__(self, y, h, Hty, H1, eta, kappa, lambda_ ,delta, xmin, xmax, phi, x, xstar, Nx, Ny, Nz, NbIt,
                 cores_number, timemax, setting=None):
        self.x = x
        self.y = y
        self.h = h
        self.Hty = Hty
        self.H1 = H1
        self.lambda_ = lambda_
        self.delta = delta
        self.kappa = kappa
        self.eta=eta
        self.phi = phi
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.xstar = xstar
        self.xmin = xmin
        self.xmax = xmax
        self.NbIt = NbIt
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.timemax = timemax
        self.stop = 1e-5
        self.modaff = 1
        self.Crit = []
        self.Time = []
        self.NormX = []
        self.Ndx = []
        self.SNR = []
        self.Err = []
        self.Mem = []
        self.Path = []
        self.setting = setting

        if cores_number == 0:
            self.cores_number = mp.cpu_count()
        else:
            self.cores_number = cores_number

        self.blocklist = set_blocks(self.cores_number-1, self.Nz)

        """ == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
        MM - MG Algorithm, Version 1.0
        Kindly report any suggestions or corrections to emilie.chouzenoux @ univ - mlv.fr
        ----------------------------------------------------------------------
        
        Input: 
        y: the degraded data 
        H: the linear degradation operator(function handle)
        H_adj: the adjoint of the linear degradation operator % (function handle)
        tau ,lambda ,delta: the regularization parameters
        xmin, xmax: the bounds
        phi: the penalty function flag as indicated below
        % (1)
        phi(u) = (1 - exp(-u. ^ 2. / (2 * delta ^ 2)));
        % (2)
        phi(u) = (u. ^ 2). / (2 * delta ^ 2 + u. ^ 2);
        % (3)
        phi(u) = log(1 + (u. ^ 2). / (delta ^ 2));
        % (4)
        phi(u) = sqrt(1 + u ^ 2 / delta ^ 2) - 1;
        % (5)
        phi(u) = 1 / 2
        u ^ 2;
        NbIt: the max number of iterations.
        
        Output:
        x: the restored image
        Crit: the values of the criterion along minimization process
        NGrad: the value of gradient norm along minimization process
            
         Minimization of F(x) = sum_z( 1/2 || H(x) - y ||**2 + lambda sum(phi(V(x))) + tau | | x | | ^ 2 + 1/2 eta | | Proj_[xmin;xmax]-x | | ^ 2 )
        """
    def optimize(self):

        print('****************************************')
        print('Majorize-Minimize Memory Gradient Algorithm')
        print('-> SPMD BLOCK VERSION (MPI) <-')
        print('CORES NUMBER = ', self.cores_number)

        if self.phi == 1:
            print('phi(u) =  (1-exp(-u^2/(2*delta^2)))')
        elif self.phi == 2:
            print('phi(u) = (u^2)/(2*delta^2 + u^2)')
        elif self.phi == 3:
            print('phi(u) = log(1 + (u^2)/(delta^2))')
        elif self.phi == 4:
            print('phi(u) =  sqrt(1 + u^2/delta^2)-1')
        elif self.phi == 5:
            print('phi(u) = 1/2 u^2')

        print('lambda = ', self.lambda_ , ', delta = ', self.delta, ', eta = ', self.eta, ' and kappa = ', self.kappa)
        print('xmin = ', self.xmin, ' and xmax = ', self.xmax)

        #Hop = lambda x : apply_PSFvar3D(x, h)
        #Hop_adj = lambda y : apply_PSFadjvar3D(y, h)

        #self.Time.append(time.time())
        self.Ndx.append(np.inf)
        self.NormX.append(np.linalg.norm(self.x-self.xstar))
        #dx = self.x

        #creating pool worker and initializing shared array
        #self.X = mp.RawArray('d',self.Nx*self.Ny*self.Nz)
        #self.X_np = np.frombuffer(self.X).reshape(self.x.shape)
        #np.copyto(self.X_np, self.x)

        #pool = mp.Pool(processes=self.cores_number, initargs=(self.X))
        #start = time.time()
        #self.critz = [pool.apply(ComputeCriterionPar, args=(z, self.X_np, self.h, self.y, self.phi, self.eta, self.lambda_, self.delta, self.kappa, self.xmin, self.xmax, self.Nx, self.Ny, self.Nz)) for z in range(self.Nz)]
        #pool.close()
        #print('elapsed time', time.time() - start)

        #self.Crit.append(np.sum(self.critz))
        #self.Time.append(0)
        #print('Initial criterion value = ', self.Crit[-1])


    # Start Master and Workers
        self.connec = mp.Pipe()
        Master = PAR3MG_Master_worker(self.y, self.h, self.Hty, self.H1, self.eta, self.kappa, self.lambda_ , self.delta, self.xmin, self.xmax, self.phi, self.x, self.xstar, self.Nx, self.Ny, self.Nz, self.NbIt, self.timemax, self.cores_number, self.blocklist, self.connec[0], self.setting)
        Master.start()
        os.system("taskset -p -c% d% d" % ((0), Master.pid))
        #for process in psutil.process_iter():
        #    if process.pid == Master.pid:
        #        process.cpu_affinity([0])

        x_final, dx, Crit_final, Time_final, Error, SNR = self.connec[1].recv()
        Master.terminate()
        self.Crit = Crit_final
        self.x = x_final
        self.dx = dx
        self.Time = Time_final
        self.Error = Error
        self.SNRs = SNR
        #self.Time = Master.Time

        #plt.plot(self.Crit,label='python')
        #plt.plot(loadmat('/home/mathieuchalvidal/Downloads/bp3mg/Critblkmpi.mat')['Critblkmpi'].T,label='matlab')
        #plt.legend()
        #plt.title('Evolution of F(xk) under Matlab and Python for {} iterations with {} workers'.format(self.NbIt,self.cores_number-1))
        #plt.savefig('Aneurysm2_64x64x24_6workers')
        #plt.show()

        print('Iteration number = ', len(self.Crit))
        print('Computation time (cpu) =', self.Time[-1])
        # print('Memory =', self.Mem[-1],' MB')
        print('Final criterion value = ', np.sum(self.Crit[-1]))
        #print('Final SNR value = ', SNRend)
        print('****************************************')
        return