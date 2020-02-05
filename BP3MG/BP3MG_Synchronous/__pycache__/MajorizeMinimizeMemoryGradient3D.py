import time
import numpy as np
from BP3MG_Standard.Critere3D import Critere3D
from BP3MG_Standard.Voperator3D import *
import matplotlib.pyplot as plt

class MajorizeMinimizeMemoryGradient3D:

    def __init__(self, y, H, H_adj, eta, lambda_, delta, kappa, phi, x, xstar, xbar, xmin, xmax, NbIt, timemax):

        """
        :param y: observed data
        :param H: Gaussian blur operator
        :param H_adj: adjoint of the gaussian blur operator
        :param eta: regularization parameter on the distance to the hypercube [xmin, xmax]
        :param lambda_: regularization parameter on the horizontal total variation norm
        :param delta: second regularization parameter on the horizontal total variation norm
        :param kappa: regularization parameter on the vertical total variation norm
        :param phi: choice of regularization function
        :param x: estimation
        :param xstar: ground truth
        :param xbar: minimal estimation
        :param xmin: lower bound on image pixel value
        :param xmax: upper bound on image pixel value
        :param NbIt: nb of iteration for the algorithm to reach
        :param timemax: maximal time of computation
        :return:
        """

        self.y = y
        self.H = H
        self.H_adj = H_adj
        self.eta = eta
        self.lambda_ = lambda_
        self.delta = delta
        self.kappa = kappa
        self.phi = phi
        self.x = x
        self.xstar = xstar
        self.xbar = xbar
        self.xmin = xmin
        self.xmax = xmax
        self.NbIt = NbIt
        self.timemax = timemax

        self.Crit = []
        self.Time = []
        self.NormX = []
        self.Ndx = []
        self.SNR = []
        self.Err = []
        self.Mem = []

        self.Nx, self.Ny, self.Nz = x.shape
        self.stop = 1e-5
        self.modaff = 1 # Frequency of display
        # dispfig = floor([Nz / 3 Nz / 2 2 * Nz / 3]); # [Nz / 4 Nz / 2 3 * Nz / 4]; % images to display
        # h = plt.imshow(dispfig);
        # close(h);

        """ 
        Printing optimization parameters and regularisation functions
        """

        print('****************************************')
        print('Majorize-Minimize Memory Gradient Algorithm')
        print('-> STANDARD VERSION <-')
        phixy = phi[0]
        phiz = phi[1]

        if phixy == 1:
            print('phixy(u) =  (1-exp(-u^2/(2*delta^2)))')
        elif phixy == 2:
            print('phixy(u) = (u^2)/(2*delta^2 + u^2)')
        elif phixy == 3:
            print('phixy(u) = log(1 + (u^2)/(delta^2))')
        elif phixy == 4:
            print('phixy(u) =  sqrt(1 + u^2/delta^2)-1')
        elif phixy == 5:
            print('phixy(u) = 1/2 u^2')
        elif phixy == 6:
            print('phixy(u) = 1-exp(-((1 + (u^2)/(delta^2))^(1/2)-1))')
        elif phixy == 7:
            print('phixy(u) =  (1 + u^2/delta^2)^(1/4)-1')

        if phiz == 1:
            print('phiz(u) =  (1-exp(-u^2/(2*delta^2)))')
        elif phiz == 2:
            print('phiz(u) = (u^2)/(2*delta^2 + u^2)')
        elif phiz == 3:
            print('phiz(u) = log(1 + (u^2)/(delta^2))')
        elif phiz == 4:
            print('phiz(u) =  sqrt(1 + u^2/delta^2)-1')
        elif phiz == 5:
            print('phiz(u) = 1/2 u^2')

        print('lambda = ', lambda_)
        print('delta = ', delta)
        print('kappa = ', kappa)
        print('eta = ', eta)
        print('xmin = ', xmin, ' and xmax = ', xmax)

    def optimize(self):

        """
        Starting computation
        """
        x = self.x
        xstar = self.xstar
        xbar = self.xbar

        Cri, Grad = Critere3D(x, self.y, self.H, self.H_adj, self.eta, self.lambda_ , self.kappa, self.delta, self.phi, self.xmin, self.xmax)
        self.Crit.append(Cri)

        print('Initial criterion value = ',Cri)

        self.Time.append(time.time())
        self.NormX.append(np.linalg.norm(x-xstar))
        self.Ndx.append(np.inf)

        if xbar is None:
            self.SNR.append(np.Inf)
        else:
            self.Err.append(np.linalg.norm(x-xbar)**2/(np.linalg.norm(xbar)**2))
            self.SNR.append(10*np.log10(1/self.Err[-1]))

        for k in range(self.NbIt):

            #Stopping criteria
            if self.Ndx[k] < self.stop:
                break

            if k % self.modaff == 0:
                print('to be plotted: iteration n°:{}'.format(k))
                #subplot(121)
                #imagesc(y(:,:, floor(Nz / 2)), [0 1]); axis
                #subplot(122)
                #imagesc(x(:,:, floor(Nz / 2))); axi

            Vvg, Vhg, Vtg = Voperator3D(Grad)
            Hg = self.H(Grad)

            if k == 0:

                B = self.majorante1(x, -Grad, Vvg, Vhg, Vtg, Hg, self.phi, self.eta, self.lambda_, self.kappa, self.delta, self.xmin, self.xmax)
                s = np.sum(Grad**2)/B
                dx = s*-Grad
                Hdx = -s*Hg
                Vvdx = -s*Vvg
                Vhdx = -s*Vhg
                Vtdx = -s*Vtg


            #MEMORY
            # vv = whos;
            # mem = 0;
            # for j=1:length(vv)
            # mem = mem + vv(j).bytes;
            # end
            # Mem0 = mem * 1e-6;
            # disp(['   Memory before clear =', num2str(Mem0), ' MB']);

            #clear
            #Grad, clear
            #Vvg, clear
            #Vhg, clear
            #Vtg, clear
            #Hg, clear
            #B

            else:

                B = self.majorante2(x, -Grad, dx, -Vvg, -Vhg, -Vtg, Vvdx, Vhdx, Vtdx, -Hg, Hdx, self.phi, self.eta, self.lambda_ , self.kappa, self.delta, self.xmin, self.xmax)
                d1 = -np.sum(Grad**2)
                d2 = np.sum(dx*Grad)
                s = -np.linalg.pinv(B)@np.array([d1,d2])

                dx = s[0]*(-Grad) + s[1]*dx

                Hdx = -s[0]*Hg + s[1]*Hdx
                Vvdx = -s[0]*Vvg + s[1]*Vvdx
                Vhdx = -s[0]*Vhg + s[1]*Vhdx
                Vtdx = -s[0]*Vtg + s[1]*Vtdx

                """
                MEMORY
                % vv = whos;
                % mem = 0;
                % for j=1:length(vv)
                % mem = mem + vv(j).bytes;
                % end
                % Mem0 = mem * 1e-6;
                % disp(['   Memory before clear =', num2str(Mem0), ' MB']);
    
                clear
                Vvg, clear
                Vhg, clear
                Vtg, clear
                Hg
                end
                """

            # update, critere and error computation, time storage
            x = x + dx

            Cri, Grad = Critere3D(x, self.y, self.H, self.H_adj, self.eta, self.lambda_ , self.kappa, self.delta, self.phi, self.xmin, self.xmax)
            self.Crit.append(Cri)


            self.Time.append(time.time())

            if xbar is not None:
                self.Err.append(np.linalg.norm(x-xbar)**2/(np.linalg.norm(xbar)**2))
                self.SNR.append(10*np.log10(1/self.Err[-1]))

            self.NormX.append(np.linalg.norm(x-xstar))
            self.Ndx.append(np.linalg.norm(dx)/np.linalg.norm(x))

            """MEMORY
            % vv = whos;
            % mem = 0;
            % for j=1:length(vv)
            % mem = mem + vv(j).bytes;
            % end"""
            #Mem(k) = 0; % mem * 1e-6;

            if k % self.modaff == 0:
                print('---')
                print('Iteration number = ', k)
                print('Criterion value = ', self.Crit[-1])
                print('Error to ground truth', self.Err[-1])
                print('Norm(dx) = ',self.Ndx[-1])
                print('Computation time = ', self.Time[-1]-self.Time[0])
                #plt.imshow(x[:,:,30])
                #plt.show()

                if xbar is not None:
                    print('SNR value = ', self.SNR[-1])

            if self.Time[-1]-self.Time[0] > self.timemax:
                print('MAXIMAL TIME!!')
                break

            plt.imshow(x[:,:,10])
            plt.show()

        SNRend = self.SNR[-1]
        print('Iteration number = ', len(self.Crit))
        print('Computation time (cpu) =', self.Time[-1]-self.Time[0])
        #print('Memory =', self.Mem[-1],' MB')
        print('Final criterion value = ', self.Crit[-1])
        print('Final SNR value = ', SNRend)
        print('****************************************')

        return SNRend, self.SNR, x, self.Crit, self.Ndx, self.Time, self.Mem



    def majorante1(self, x, d1, Vvd1, Vhd1, Vtd1, Hd1, phi, eta, lambda_ , kappa, delta, xmin, xmax):

        phiXY = phi[0]
        phiZ = phi[1]

        Vvx, Vhx, Vtx = Voperator3D(x)

        if phiXY == 1:
            wXY_Vx = (1/(delta**2))*np.exp(-((np.sqrt(Vvx**2 + Vhx**2))**2)/(2*delta**2))
        elif phiXY == 2:
            wXY_Vx = (4*delta**2)/(2*delta**2+(np.sqrt(Vvx**2 + Vhx**2))**2)**2
        elif phiXY == 3:
            wXY_Vx = 2/(delta**2 + (np.sqrt(Vvx**2 + Vhx**2))**2)
        elif phiXY == 4:
            wXY_Vx = (1/(delta**2))*(1 + ((np.sqrt(Vvx**2 + Vhx**2))**2)/delta**2)**(-1/2)
        elif phiXY == 5:
            wXY_Vx = np.diag(np.sqrt(Vvx**2 + Vhx**2))
        elif phiXY == 6:
            wXY_Vx = (1/(delta**2))*((1+((np.sqrt(Vvx**2 + Vhx**2))**2)/delta**2)**(-1 / 2))*np.exp(-((1+((np.sqrt(Vvx**2 + Vhx**2))**2)/(delta**2))**(1/2)-1))
        elif phiXY == 7:
            pow = 0.25
            wXY_Vx = ((2*pow)/(delta**2))*(1+((np.sqrt(Vvx**2 + Vhx**2))**2)/delta**2)**(pow-1)


        p = 0.1
        if phiZ == 1:
            wZ_Vx = (1/(p**2))*np.exp(-(Vtx**2)/((2**p)**2))
        elif phiZ == 2:
            wZ_Vx = ((4**p)**2)/((2**p)**2+Vtx**2)**2
        elif phiZ == 3:
            wZ_Vx = 2/(p**2 + Vtx**2)
        elif phiZ == 4:
            wZ_Vx = (1/((p**2)*(1+(Vtx**2)/((p**2)**(-1/2)))))
        elif phiZ == 5:
            wZ_Vx = np.ones(Vtx.shape)

        """
        clear
        #Vvx, clear
        #Vhx, clear
        #Vtx
        """

        B = np.sum(Hd1**2) + self.lambda_*np.sum(Vvd1*(wXY_Vx*Vvd1)+Vhd1*(wXY_Vx*Vhd1)) + self.kappa*np.sum(Vtd1*(wZ_Vx*Vtd1)) + self.eta*np.sum(d1[(x <= xmin) + (x >= xmax)]**2)
        
        """
        % MEMORY
        % vv = whos;
        % memM = 0;
        % for j=1:length(vv)
        % memM = memM + vv(j).bytes;
        % end
        % MemM(k) = memM * 1e-6;
        % disp(['   Memory majorantem =', num2str(MemM(k)), ' MB']);
        end"""
            
        return B

    def majorante2(self, x, d1, d2, Vvd1, Vhd1, Vtd1, Vvd2, Vhd2, Vtd2, Hd1, Hd2, phi, eta, lambda_, kappa, delta, xmin, xmax):

        phiXY = phi[0]
        phiZ = phi[1]
        Vvx, Vhx, Vtx = Voperator3D(x)

        if phiXY == 1:
            wXY_Vx = (1/(delta**2))*np.exp(-((np.sqrt(Vvx**2 + Vhx**2))**2)/(2*delta**2))
        elif phiXY == 2:
            wXY_Vx = (4*delta**2)/(2*delta**2+(np.sqrt(Vvx**2 + Vhx**2))**2)**2
        elif phiXY == 3:
            wXY_Vx = 2/(delta**2 + (np.sqrt(Vvx**2 + Vhx**2))**2)
        elif phiXY == 4:
            wXY_Vx = (1/(delta**2))*(1 + ((np.sqrt(Vvx**2 + Vhx**2))**2)/delta**2)**(-1/2)
        elif phiXY == 5:
            wXY_Vx = np.diag(np.sqrt(Vvx**2 + Vhx**2))
        elif phiXY == 6:
            wXY_Vx = (1/(delta**2))*((1+((np.sqrt(Vvx**2 + Vhx**2))**2)/delta**2)**(-1 / 2))*np.exp(-((1+((np.sqrt(Vvx**2 + Vhx**2))**2)/(delta**2))**(1/2)-1))
        elif phiXY == 7:
            pow = 0.25
            wXY_Vx = ((2*pow)/(delta**2))*(1+((np.sqrt(Vvx**2 + Vhx**2))**2)/delta**2)**(pow-1)

        p = 0.1
        if phiZ == 1:
            wZ_Vx = (1/(p**2))*np.exp(-(Vtx**2)/((2**p)**2))
        elif phiZ == 2:
            wZ_Vx = ((4**p)**2)/((2**p)**2+Vtx**2)**2
        elif phiZ == 3:
            wZ_Vx = 2/(p**2 + Vtx**2)
        elif phiZ == 4:
            wZ_Vx = (1/((p**2))*(1+(Vtx**2)/((p**2)**(-1/2))))
        elif phiZ == 5:
            wZ_Vx = np.ones(Vtx.shape)
                                
        """
        clear
        Vvx, clear
        Vhx, clear
        Vtx
        """
        
        d1tVtWVd1 = self.lambda_*np.sum(Vvd1*(wXY_Vx*Vvd1)+Vhd1*(wXY_Vx*Vhd1))+ self.kappa*np.sum(Vtd1*(wZ_Vx*Vtd1))
        d1tVtWVd2 = self.lambda_*np.sum(Vvd1*(wXY_Vx*Vvd2)+Vhd1*(wXY_Vx*Vhd2))+ self.kappa*np.sum(Vtd1*(wZ_Vx*Vtd2))
        d2tVtWVd2 = self.lambda_*np.sum(Vvd2*(wXY_Vx*Vvd2)+Vhd2*(wXY_Vx*Vhd2))+ self.kappa*np.sum(Vtd2*(wZ_Vx*Vtd2))
        
        """
        clear
        wXY_Vx, clear
        wZ_Vx, clear
        Vvd1, clear
        Vvd2, clear
        Vhd1, clear
        Vhd2, clear
        Vtd1, clear
        Vtd2
        """
        d1_min = np.where((x<=xmin),d1,0)
        d2_min = np.where((x <= xmin), d2, 0)
        d1_max = np.where((x >= xmax), d1, 0)
        d2_max = np.where((x >= xmax), d2, 0)

        #d1UNmind1 = d1_min**2
        #d1UNmaxd1 = d1_max**2
        #d1UNmind2 = d1_min*d2_min
        #d1UNmaxd2 = d1_max*d2_max
        #d2UNmind2 = d2_min**2
        #d2UNmaxd2 = d2_max**2


        B11 = np.sum(Hd1**2) + d1tVtWVd1 + self.eta*np.sum(d1_min**2 + d1_max**2)
        B12 = np.sum(Hd1*Hd2) + d1tVtWVd2 + self.eta*np.sum(d1_min*d2_min + d1_max*d2_max)
        B22 = np.sum(Hd2**2) + d2tVtWVd2 + self.eta*np.sum(d2_min**2 + d2_max**2)
        B = np.array([[B11, B12],[B12, B22]])

        
        """
        % MEMORY
        % vv = whos;
        % memM = 0;
        % for j=1:length(vv)
        % memM = memM + vv(j).bytes;
        % end
        % MemM(k) = memM * 1e-6;
        % disp(['   Memory majorantem =', num2str(MemM(k)), ' MB'])
        """

        return B