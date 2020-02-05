import numpy as np
import os
import time
import multiprocessing as mp
from BP3MG.ComputeCriterionPar import ComputeCriterionPar
from BP3MG.LOOP3MGpar import LOOP3MGpar


class PAR3MG_Master_worker(mp.Process):

    def __init__(self, y, h, Hty, H1, eta, kappa, lambda_ ,delta, xmin, xmax, phi, x, xstar, Nx, Ny, Nz, NbIt, timemax, num_workers, blocklist, connection, setting):

        mp.Process.__init__(self)
        self.x = x
        self.dx = np.zeros(self.x.shape)
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
        self.H = [self.h(z) for z in range(self.Nz)]
        self.xstar = xstar
        self.xmin = xmin
        self.xmax = xmax
        self.NbIt = NbIt
        self.timemax = timemax
        self.stop = 1e-5
        self.modaff = 1
        self.critz = np.zeros(Nz)
        self.Crit = []
        self.Time = []
        self.Timesending = []
        self.NormX = []
        self.Ndx = []
        self.SNR = []
        self.Err = []
        self.Mem = []
        self.num_workers = num_workers
        self.p3 = int((self.h(0).shape[2]-1)/2)
        self.z_select = np.zeros(self.num_workers-1).astype(int)
        self.init = np.ones(self.Nz)
        self.blocklist = blocklist
        self.connec = connection
        self.connections = [mp.Pipe() for i in range(num_workers)]
        self.cpu_per_cent = []
        self.Workers = [PAR3MG_slave_worker(i, self.connections[i][1], x, self.y, self.Hty, self.H1, h, self.p3, self.lambda_, self.delta, self.kappa, self.eta, self.phi, self.Nx, self.Ny, self.Nz, self.xmin, self.xmax, setting) for i in range(num_workers - 1)]
        for l,w in enumerate(self.Workers):
            w.start()
            os.system("taskset -p -c %d %d" % ((l%os.cpu_count()+1), w.pid))
        self.PIDs = [worker.pid for worker in self.Workers]
        print('PID of workers : ',self.PIDs)


    def run(self):

        pool = mp.Pool(processes=self.num_workers)
        self.critz = [pool.apply(ComputeCriterionPar, args=(z, self.x, self.H, self.y, self.phi, self.eta, self.lambda_, self.delta, self.kappa, self.xmin, self.xmax, self.Nx, self.Ny, self.Nz)) for z in range(self.Nz)]
        pool.close()
        self.Crit.append(np.sum(self.critz))
        self.SNR.append(10 * np.log10(np.sum(self.x ** 2) / np.sum((self.x.flatten() - self.y.flatten()) ** 2)))
        print('Initial criterion value = ', self.Crit[-1])
        self.Time.append(0)

        for k in range(1,self.NbIt+1):

            # ENVOI DES VECTEURS SYNCHRONISES ET PRET A RECEVOIR

            self.start = time.time()

            for c in range(self.num_workers-1):

                Blk = self.blocklist[c]
                lb = len(Blk)
                idx = k%lb
                self.sending_update(self.connections[c][0], Blk[idx], self.init[Blk[idx]], self.p3)
                #Computation by workers at this point

            self.Timesending.append(time.time() - self.start)

            for c in range(self.num_workers - 1):

                xloc, dxloc, z, critz = self.connections[c][0].recv()
                self.x[:,:,z] = xloc.reshape((self.Nx,self.Ny))
                self.dx[:,:,z] = dxloc.reshape((self.Nx, self.Ny))
                self.critz[z] = critz
                self.init[z] = 1

            self.Time.append(time.time()-self.start)
            self.Crit.append(np.sum(self.critz))
            print('Criterion value = ', self.Crit[-1])
            self.Ndx.append(np.linalg.norm(self.dx)/np.linalg.norm(self.x))
            self.NormX.append(np.linalg.norm(self.x - self.xstar))
            self.SNR.append(10 * np.log10(np.sum(self.x ** 2) / np.sum((self.x.flatten() - self.y.flatten()) ** 2)))
            print('Norm(dx)/Norm(x) = ', self.Ndx[-1])
            print('Error f(xk) - f* = ', self.NormX[-1])

            if self.Ndx[-1] < self.stop:
                print('STOPPING CRITERION REACHED!')
                break
            if time.time() - self.start > self.timemax:
                print('MAXIMAL TIME!!')
                break

        for w in self.Workers:
            w.terminate()

        self.connec.send((self.x, self.dx, self.Crit, self.Time, self.NormX, self.SNR))

    def sending_update(self, connec, z, init, p3):
        package = (self.x[:,:,max(0,z-2*p3):min(self.Nz,z+2*p3+1)].copy(),
                   self.dx[:,:,z].copy(),
                   z,
                   init)
        connec.send(package)


class PAR3MG_slave_worker(mp.Process):

    def __init__(self,c , connection, x, y, Hty, H1, h, p3, lambda_, delta, kappa, eta, phi, Nx, Ny, Nz, xmin, xmax, setting):
        mp.Process.__init__(self)
        self.x = x
        self.c= c
        self.dx= np.zeros((Nx,Ny,Nz))
        self.y = y
        self.Hty = Hty
        self.H1 = H1
        self.h = h
        self.p3 = p3
        self.lambda_ = lambda_
        self.delta = delta
        self.kappa = kappa
        self.eta = eta
        self.phi = phi
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.H = [self.h(z) for z in range(self.Nz)]
        self.xmin = xmin
        self.xmax = xmax
        self.setting = setting
        self.connection = connection

    def run(self):
        while True:
            x_share_z, dx_share_z, z, init_z = self.connection.recv()
            if self.setting == 'uniform_1_sec':
                time.sleep(np.random.uniform(0,1))
            elif self.setting == 'one_1_sec':
                if self.c == 3:
                    time.sleep(np.random.uniform(0,1))
            elif self.setting == '3_at_0.25_0.5_1_sec':
                if self.c%3==0:
                    time.sleep(np.random.uniform(0,1))
                elif self.c%3==1:
                    time.sleep(np.random.uniform(0,0.5))
                elif self.c%3==2:
                    time.sleep(np.random.uniform(0,0.25))

            list_n3 = np.arange(max(0,z-2*self.p3),min(self.Nz,z+2*self.p3+1))
            xloc, dxloc = LOOP3MGpar(z, init_z, x_share_z, list_n3, dx_share_z, self.Hty[:,:,z], self.H1[:,:,z], self.H, self.Nx, self.Ny, self.Nz, self.eta, self.lambda_ , self.delta, self.kappa, self.phi, self.xmin, self.xmax, self.p3)
            self.x[:, :, list_n3] = x_share_z
            self.dx[:, :, z] = dx_share_z
            self.x[:, :, z] = xloc.reshape((self.Nx, self.Ny))
            self.dx[:, :, z] = dxloc.reshape((self.Nx, self.Ny))
            critz = ComputeCriterionPar(z, self.x, self.H, self.y, self.phi, self.eta, self.lambda_, self.delta, self.kappa, self.xmin, self.xmax, self.Nx, self.Ny, self.Nz)
            self.connection.send((xloc, dxloc, z, critz))



