import numpy as np

def elementwiseThresholdS( kappa, v ):
    res = np.zeros_like(v)
    res[v>kappa] = v[v>kappa]-kappa
    res[v < -kappa] = v[v < -kappa]+kappa
    return res

# def differenceMatrix(n):
#     D = np.diag(np.ones((n,))) - np.diag(0.5*np.ones((n-1,)),1) - np.diag(0.5*np.ones((n-1,)),-1)
#     D[0,1] = -1
#     D[-1,-2] = -1
#     return D

def differenceMatrix(n):
    D = np.diag(np.ones((n,))) - np.diag(np.ones((n-1,)),-1)
#     D[0,:]=0
    return D
    
def ADMM(C,b,lam,rho,
         x0=None,
         maxIterations = 10.):
    Nb,Nx = C.shape
    D = differenceMatrix(Nx)
    
    rhoDt = rho*D.T
    rhoDtD = rhoDt.dot(D)
    
    CtC = (C.T).dot(C)
    Ctb = (C.T).dot( b )
    
    inverseMatrix = np.linalg.inv( CtC + rhoDtD )
    
    iMCtb = inverseMatrix.dot(Ctb)
    iMrhoDt = inverseMatrix.dot(rhoDt)
    
    xlsq = np.linalg.inv(CtC).dot( Ctb )
    
    
    if x0 is None:
        x0 = Ctb
    x = x0
    z = D.dot(x0)
    u = 0*x0
    
    for idx in range(maxIterations):
        v = z - u
        x = iMCtb + iMrhoDt.dot( v )
        Dx = D.dot(x)
        z = elementwiseThresholdS( lam/rho , Dx+u )
        u = u + Dx - z
        
    return x
    