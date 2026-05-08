import numpy as np
import torch
from scipy.sparse import coo_matrix
import graph_tool.all as gt
from scipy.stats import genpareto
import random

def generate_random_sparse_tensor(size, num_non_zeros, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if num_non_zeros > size * size:
        raise ValueError("Number of non-zeros cannot be greater than the total number of elements in the matrix")

    indices = np.random.choice(size * size, int(num_non_zeros), replace=False)
    row_indices, col_indices = np.unravel_index(indices, (size, size))
    data = np.ones(num_non_zeros)

    sparse_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(size, size))
    return torch.tensor(sparse_matrix.todense(), dtype=torch.float)

def degree_rv(dist, gamma=None, a=None, beta=None, mean_k=None, N=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if dist == 'sf':
        assert gamma is not None, "gamma must be specified for SF distribution"
        assert a is not None, "mu must be specified for SF distribution"
        return genpareto.rvs(c=1/(gamma-1), scale=a/(gamma-1), loc=a, random_state=seed)
    elif dist == 'binom':
        assert mean_k is not None, "mean_k must be specified for binom distribution"
        assert N is not None, "N must be specified for binom distribution"
        return np.random.binomial(n=N, p=mean_k/N)
    elif dist == 'exp':
        assert beta is not None, "N must be specified for binom distribution"
        return np.random.exponential(scale=beta)

def generate_random_directed_adjacency(graph_size, in_dist, out_dist, gamma=None, a=None, beta=None, mean_k=None, seed=None):
    if seed is not None:
        gt.seed_rng(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    g = gt.random_graph(graph_size,
                        lambda _: (degree_rv(dist=in_dist, gamma=gamma, a=a, beta=beta, mean_k=mean_k, N=graph_size, seed=seed),
                                   degree_rv(dist=out_dist, gamma=gamma, a=a, beta=beta, mean_k=mean_k, N=graph_size, seed=seed)),
                        directed=True)

    # Convert the graph to an adjacency matrix
    adjacency_matrix = gt.adjacency(g).toarray().astype(np.int16)

    return torch.tensor(adjacency_matrix, dtype=torch.float)


def reduce_mask(mask, input_dim, out_dim):
    assert input_dim <= mask.shape[0] and out_dim <= mask.shape[0] ##reducing the number of rows/columnds
    return mask[:input_dim, :out_dim].clone().detach() # return a matrix with shape [input_dim x out_dim]  

