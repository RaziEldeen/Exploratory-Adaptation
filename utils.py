import numpy as np


def stress(b, x,target = 0):
    return np.abs(np.dot(b,x) - target)
                  
def Mismatch(s, M_0, mu, eps):
    return (M_0/2) * (1 + np.tanh((s - eps)/mu))

#participation ratio
def comp_d(X,d_sim,start_idx=0):
    X = X[start_idx:d_sim]
    X_bar = X.mean(0)
    X_ = X - X_bar
    C = X_.T.dot(X_)/(d_sim - start_idx + 1)
    return np.trace(C)**2/np.trace(C.dot(C))

def init_b(N = 1500, c = 0.2, alpha = 100.0, g_0 = 10.0, m_b = 0.0):
    b = np.zeros(N,)
    cN=int(np.round(N*c)) #number of non-zeroes 
    idxs = np.random.permutation(range(N))[:cN] #choose cN indicies out of N at random 
    g_b = (1/g_0)*np.sqrt(alpha/cN);
    b[idxs] = m_b + g_b*np.random.randn(cN)
    
    return b

def init_J(T, g_0=10):
    active_idxs = np.where(T!=0) #the only connections to update
    N_W_n_z = np.sum(T) # number of active connections
    avg_k = np.sum(T,1).mean() # average connectivity <K>
    
    W_int = np.zeros_like(T,dtype=np.float)
    W_int[active_idxs] = (g_0/np.sqrt(avg_k))*np.random.randn(N_W_n_z,)
    return W_int

def F(J, x):
    return np.dot(J, np.tanh(x)) - x


def run_trial(T, x_0 = None, W_0=None,  N=1500, g_0=10, b_alpha=100, m_b =0, sparsity=0.2, g_w=10, target=0, D=1e-3, eps=3, mu=0.01, M_0=4, t_int=0, t_max=2000, dt=0.1, T_stop=100, tol=1e-2):
    """
    run dynamics for a given ensemble 
  
    Parameters:
    T: NXN adjacency matrix for a predefined ensemble
    N: network size
    g_0: matrix gain for the vector b initialization
    b_alpha: parameter determining the scale of phenotype fluctuations
    m_b: mean of b[i]
    sparsity (c): fraction of non-zero elements in b
    g_w:matrix gain for J initilization
    target: y*
    D: The amplitude of the random walk
    eps,mu,M_0: parameters for the Mismatch function
    t_int: initial time for the simluation
    dt: the step-size for the dynamics
    t_max: if network does not converge end simulation at t=t_max
    T_stop: if network output y~y* for T_stop time units, then network converged.
    tol: Ms(|y-y*|)< tol is regarded as success to converge

    Returns:
    is_success: 1 if converged, 0 otherwise.
  
    """
    
    
    # initialize b
    b = init_b(N=N, alpha=b_alpha, g_0=g_0,m_b=m_b)
    
    
    # initialize J
    if W_0 is not None:
        W_rec = W_0
    else:
        W_rec = init_J(T,g_0)
        
        

    
    #active indicies (used for updating weights)
    N_W_n_z = np.sum(T)
    active_idxs = np.where(T!=0)
    
    # trial params
    T_sim = round(t_max/dt)+1;

    s = np.zeros((T_sim,)) #Will hold stress at all timesteps (not essensial)
    M_s = np.zeros((T_sim,)) #Will hold mismatch function at all timesteps - useful for stopping creterion
    is_sucess = 0 #will be set to 1 in case the network converges
    
    
    # initialize x
    X = np.zeros((T_sim,N))
    if x_0 is not None:
        X[0] = x_0
    else:
        X[0] = 10 *np.random.randn(N)
        
    #run trial
    for i in range(T_sim-1):
        # calculate mismatch
        s[i] = stress(b, X[i], target=0)
        M_s[i] = Mismatch(s[i], M_0, mu, eps)
        
        #run dynamics
        X[i+1] = X[i] + dt*F(W_rec,X[i])#(np.dot(W_rec,np.tanh(X[i])) - X[i])
        
        # update weights - EA
        delta = np.sqrt(M_s[i]*dt*D)*np.random.randn(N_W_n_z,)
        W_rec[active_idxs] = W_rec[active_idxs] + delta
        
        #stopping creterion:
        if i*dt> T_stop:
            if ~np.any(M_s[i-int((T_stop/dt)):i] > tol):
                is_sucess = 1
                break
    
    return is_sucess, X, i+1
