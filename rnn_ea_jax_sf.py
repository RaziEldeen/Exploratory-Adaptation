import jax
import jax.numpy as np
from jax import lax, random, jit
import numpy as NP
from scipy.stats import genpareto,genexpon

from graph_tool.generation import random_graph
from graph_tool.spectral import adjacency
from itertools import groupby
    
    
def len_iter(items):
    return sum(1 for _ in items)


def consecutive_one(data, tol=1e-2):
    try:
        res = max(len_iter(run) for val, run in groupby(data) if val<tol)
    except:
        res = 0
    return res

def stress(b, x,target = 0):
    return np.abs(np.dot(b,x) - target)
                  
def Mismatch(s, M_0, mu, eps):
    return (M_0/2) * (1 + np.tanh((s - eps)/mu))    

    
    
def degree_rv(dist, gamma=None, a=None, beta=None, mean_k=None, N=None):
    if dist == 'sf':
        assert gamma is not None, "gamma must be specified for SF distribution"
        assert a is not None, "mu must be specified for SF distribution"
        return genpareto.rvs(c = 1/(gamma-1), scale=a/(gamma-1), loc=a)
    elif dist == 'binom':
        assert mean_k is not None, "mean_k must be specified for binom distribution"
        assert N is not None, "N must be specified for binom distribution"
        return NP.random.binomial(n=N, p=mean_k/N)
    elif dist == 'exp':
        assert beta is not None, "N must be specified for binom distribution"
        return NP.random.exponential(scale=beta)
    
    
def init_b(key, N = 1500, c = 0.2, alpha = 100.0, g_0 = 10.0, m_b = 0.0):
    b = np.zeros(N,)
    cN=np.round(N*c).astype(int) #number of non-zeroes 
    idxs = jax.random.permutation(key,np.arange(N))[:cN] #choose cN indicies out of N at random 
    ##    idxs = jax.random.choice(key, N, shape = (cN,), replace=False)
    g_b = (1/g_0)*np.sqrt(alpha/cN);
    x = b.at[idxs].set(m_b + g_b*jax.random.normal(key,shape=(cN,)))
    return x

def init_J(key, T, g_0=10):
    active_idxs = np.where(T!=0) #the only connections to update
    N_W_n_z = np.sum(T).astype(int) # number of active connections
    avg_k = np.sum(T,1).mean() # average connectivity <K>
    
    W_int = np.zeros_like(T,dtype=float)
    J = W_int.at[active_idxs].set((g_0/np.sqrt(avg_k))*jax.random.normal(key, shape = (N_W_n_z,)))
    return J
@jit
def F(J, x):
    return np.dot(J, np.tanh(x)) - x

@jit
def comp_d(X):
    m = X.shape[0]
    X_ = X - X.mean(0)
    C = X_.T.dot(X_)/m
    return np.trace(C)**2/np.trace(C.dot(C))
    
    
def run_dynamics_all(key,n_trials=200, dt=0.1,g_0 = 10 ,t_max=2000, gamma= 2.4, beta = 3.5, N=500,alpha = 0.6, a=1):
    ds = []
    n_sim = np.round(t_max/dt).astype(int)
    inp = np.zeros((n_sim,1))
    #params for first figure
    for i in range(n_trials):
        key, Jkey, hubkey, strengthkey = jax.random.split(key,4)
        g = random_graph(N-1, deg_sampler=lambda:(degree_rv(dist='binom',gamma=gamma,beta=beta, a=a,mean_k=mean_k,N=N-1)
                                                 ,degree_rv(dist='binom',gamma=gamma,beta=beta, a=a,mean_k=mean_k,N=N-1)))
            #new_PR[(sigma,alpha,N)] = []
        T = np.array(adjacency(g).todense()).astype(np.uint32)

        hub_in = np.zeros((N-1,))
        hub_out_w = np.zeros((N,))

        # Randomly pick alpha*N nodes to connect to the hub 
        hub_out = jax.random.choice(hubkey, a=2,shape=(N,), p = np.array([1-alpha,alpha]))

        J = init_J(Jkey,T,g_0=10) 
        active_idxs = np.where(hub_out!=0)[0] #indices of hubs neihbour
        hub_out_w = hub_out_w.at[active_idxs].set(sigma*jax.random.normal(strengthkey,(hub_out.sum(),)))


        # add hub to the adjacency&weights matrix
        #T_new = np.row_stack((T, hub_in)) #add a row of zeros corresoponding to the in neigbours of the added hub
        #T_new = np.column_stack((T_new,hub_out)).astype(np.uint)# add the hub as the last column 

        #similarly, add hub to the weight matrix
        W = np.row_stack((J, hub_in))
        W = np.column_stack((W,hub_out_w))


        def f(x,inp):
            x = x
            x_ = x + dt*F(W,x)
            return x_,x_


        def run(h0,inp):
            _,state_hist = lax.scan(f,h0,inp)
            return state_hist
        jit_run = jit(run)

        x0 = 10*jax.random.normal(key, shape=(N,))


        X = jit_run(x0,inp).block_until_ready()
        ds.append(comp_d(X))
    return ds

def run_dynamics_all_sf(key, mean_k=7.5,n_trials=200, dt=0.1,g_0 = 10 ,t_max=2000, gamma= 2.4, beta = 3.5, N=500,alpha = 0.6, a=1):
    ds = []
    n_sim = np.round(t_max/dt).astype(int)
    inp = np.zeros((n_sim,1))
    #params for first figure
    for i in range(n_trials):
        key, Jkey, hubkey, strengthkey = jax.random.split(key,4)
        g = random_graph(N, deg_sampler=lambda:(degree_rv(dist='sf',gamma=gamma,beta=beta, a=a,mean_k=mean_k,N=N)
                                                 ,degree_rv(dist='sf',gamma=gamma,beta=beta, a=a,mean_k=mean_k,N=N)))
            #new_PR[(sigma,alpha,N)] = []
        T = np.array(adjacency(g).todense()).astype(np.uint32)

        W = init_J(Jkey,T,g_0=10) 


        def f(x,inp):
            x = x
            x_ = x + dt*F(W,x)
            return x_,x_


        def run(h0,inp):
            _,state_hist = lax.scan(f,h0,inp)
            return state_hist
        jit_run = jit(run)

        x0 = 10*jax.random.normal(key, shape=(N,))


        X = jit_run(x0,inp).block_until_ready()
        ds.append(comp_d(X))
    return ds

def run_trial(key, T, x_0 = None, W_0=None, g_0=10, b_alpha=100, m_b =0, sparsity=0.2, g_w=10, target=0, D=1e-3, eps=3, mu=0.01, M_0=4, t_int=0, t_max=2000, dt=0.1, T_stop=100, tol=1e-2):
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
    N = T.shape[0]
    key, bkey, Jkey, Xkey = random.split(key, 4)
    # initialize b
    b = init_b(key=bkey, N=N, alpha=b_alpha, g_0=g_0,m_b=m_b)
    
    
    # initialize J
    if W_0 is not None:
        W_rec = W_0
    else:
        W_rec = init_J(Jkey, T,g_0)
        
        

    
    #active indicies (used for updating weights)
    N_W_n_z = np.sum(T)
    active_idxs = np.where(T!=0)
    
    # trial params
    T_sim = round(t_max/dt)+1;

   # s = np.zeros((T_sim,)) #Will hold stress at all timesteps (not essensial)
    M_s = np.zeros((T_sim,)) #Will hold mismatch function at all timesteps - useful for stopping creterion
    is_sucess = 0 #will be set to 1 in case the network converges
    
    
    # initialize x
    X = np.zeros((T_sim,N))
    if x_0 is not None:
        X = X.at[0].set(x_0)
    else:
        X = X.at[0].set(10 *jax.random.normal(key=Xkey, shape=(N,)))
        
    key, skey = random.split(key, 2)
    rnds = random.normal(key=skey,shape= (T_sim, N_W_n_z))
    #run trial
    for i in range(T_sim-1):
        # calculate mismatch
        #key, skey = random.split(key, 2)
        s = stress(b, X[i], target=0)
        M_s = M_s.at[i].set(Mismatch(s, M_0, mu, eps))
        
        #run dynamics
        X = X.at[i+1].set(X[i] + dt*F(W_rec,X[i]))#(np.dot(W_rec,np.tanh(X[i])) - X[i])
        
        # update weights - EA
        delta = np.sqrt(M_s[i]*dt*D)*rnds[i]
        W_rec = W_rec.at[active_idxs].set(W_rec[active_idxs] + delta)
        #stopping creterion:
        if i*dt> T_stop:
            if ~np.any(M_s[i-int((T_stop/dt)):i] > tol):
                is_sucess = 1
                break

    return is_sucess

def run_trial_2(key, T,in_len =5, W_0 = None, x0 = None, g_0=10, b_alpha=100, m_b =0, sparsity=0.2, g_w=10, target=0, D=1e-3, eps=3, mu=0.01, M_0=4, t_int=0, t_max=2000, dt=0.1, T_stop=100, tol=1e-2):
    
    n_sim = np.round(t_max/dt).astype(int)
    N = T.shape[0]
    #params for first figure
    
    key, inpkey, hubkey, strengthkey, readoutkey = jax.random.split(key,5)
    active_idxs = np.where(T!=0)
    b = init_b(key=readoutkey, N=N, alpha=b_alpha, g_0=g_0,m_b=m_b)
    
    if W_0 is None:
        W_0 = init_J(key=strengthkey,T=T, g_0=g_0)

    #hub_in = np.zeros((N-1,))
    #hub_out_w = np.zeros((N,))

    # Randomly pick alpha*N nodes to connect to the hub 
    #hub_out = jax.random.choice(hubkey, a=2,shape=(N,), p = np.array([1-alpha,alpha]))
    if x0 is None:
        x0 = 10*jax.random.normal(key, shape=(N,))
        
    h0 = {'Ms':np.asarray(0.), 'x':x0, 'W':W_0}

    def f(state,inp):
        x = state['x']
        W = state['W']
        s = stress(b,x,target)
        M_s = Mismatch(s, M_0, mu, eps)
        delta = np.sqrt(M_s*dt*D)*inp
        x_ = state['x'] + dt*F(W,x)
        W_= W.at[active_idxs].add(delta)
        state_ = {'Ms':M_s, 'x':x_, 'W':W_}
        return state_,state_

    in_ = jax.random.normal(inpkey,shape=(n_sim//in_len,in_len,T.sum()))

    def run(h0,inp):
        _,state_hist = lax.scan(f,h0,inp)
        return state_hist
    jit_run = jit(run)
    
    mms = []
    for i in range(n_sim//in_len):
        X = jit_run(h0,in_[i])
        h0 = {'Ms':X['Ms'][-1], 'x':X['x'][-1], 'W':X['W'][-1]}
        mms.append(X['Ms'])
        
    return (consecutive_one(NP.hstack(mms))>=1000)*1

def run_trials_all(key, in_len=20, mean_k=7.5,n_trials=200, dt=0.1,g_0 = 10 ,t_max=2000, gamma= 2.4, beta = 3.5, N=500, a=1):
    succeses = 0
    n_sim = np.round(t_max/dt).astype(int)
    inp = np.zeros((n_sim,1))
    #params for first figure
    for i in range(n_trials):
        #logger.debug('building graph')
        key, Jkey, hubkey, strengthkey = jax.random.split(key,4)
        g = random_graph(N, deg_sampler=lambda:(degree_rv(dist='binom',gamma=gamma,beta=beta, a=a,mean_k=mean_k,N=N)
                                                 ,degree_rv(dist='sf',gamma=gamma,beta=beta, a=a,mean_k=mean_k,N=N)))
            #new_PR[(sigma,alpha,N)] = []
        #logger.debug('building matrix')
        T = np.array(adjacency(g).todense()).astype(np.uint32)

        #logger.debug('trial starts')
        is_s = run_trial_2(key, T, in_len = in_len)
        succeses = succeses + is_s
        
    return succeses