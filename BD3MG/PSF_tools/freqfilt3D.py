import numpy as np
from numpy.fft import fftn

def freqfilt3D(t,N,M,P):
    L1, L2, L3 = t.shape
    T = np.zeros((N,M,P))
    L13 = np.ceil(L1/2).astype(int) #3
    L23 = np.ceil(L2/2).astype(int) #3
    L33 = np.ceil(L3/2).astype(int) #6

    T[0:L13,0:L23,0:L33]=t[L13-1:L1,L23-1:L2,L33-1:L3]
    
    T[0:L13,0:L23,P-L33+1:P]=t[L13-1:L1,L23-1:L2,0:L33-1]
    
    T[N-L13+1:N,0:L23,0:L33]=t[0:L13-1,L23-1:L2,L33-1:L3]
    
    T[N-L13+1:N,0:L23,P-L33+1:P]=t[0:L13-1,L23-1:L2,0:L33-1]
    
    T[0:L13,M-L23+1:M,0:L33]=t[L13-1:L1,0:L23-1,L33-1:L3]
    
    T[0:L13,M-L23+1:M,P-L33+1:P]=t[L13-1:L1,0:L23-1,0:L33-1]

    T[N-L13+1:N,M-L23+1:M,0:L33]=t[0:L13-1,0:L23-1,L33-1:L3]
    
    T[N-L13+1:N,M-L23+1:M,P-L33+1:P]=t[0:L13-1,0:L23-1,0:L33-1]

    T = fftn(T)

    return T
