import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.patches import FancyArrowPatch

def sort_eig(eigvals, eigvecs):
        sorted_idx = np.argsort(eigvals)
        sorted_eigvals = eigvals[sorted_idx]
        sorted_eigvecs = eigvecs[:,sorted_idx].copy()
        return sorted_eigvals, sorted_eigvecs

def main():
    
    n1 = 50
    n2 = 20

    # Create cycle graphs G1 and G2
    G1 = nx.cycle_graph(n1)
    G2 = nx.cycle_graph(n2)

    # Calculate the Cartesian product of G1 and G2
    G = nx.cartesian_product(G1, G2)
    L1 = nx.laplacian_matrix(G1).todense()
    L2 = nx.laplacian_matrix(G2).todense()
    L = nx.laplacian_matrix(G).todense()

    # Eigenvalues and eigenvectors
    lambdas, psis = np.linalg.eig(L1)
    mus, phis = np.linalg.eig(L2)
    eigvals, eigvecs = np.linalg.eig(L)

    # Sort the eigenvalues by asending order
    sorted_lambdas, sorted_psis = sort_eig(lambdas, psis)
    sorted_mus, sorted_phis = sort_eig(mus, phis)
    sorted_eigvals, sorted_eigvecs = sort_eig(eigvals, eigvecs)

    # Theoritic eigenvalues
    sorted_eigvals_2 = sorted([mu + lambda_ for mu, lambda_ in product(sorted_mus, sorted_lambdas) ])

    # PLot eigenvalues order stats, compare it with the theoritical values. 
    x = np.arange(n1*n2)
    plt.plot(x, sorted_eigvals,'o', x, sorted_eigvals_2)
    plt.xlabel('Order Statistic')
    plt.ylabel('Eigenvalue')
    plt.legend(['L eigenvals','$\lambda_i + \mu_j$'])
    plt.show()
    
    # Plot the graph topology
    # Create positions for all nodes - 2D grid
    pos = {(i, j): (i, j) for i in range(n1) for j in range(n2)}

    # get the 1st, 2nd, 6th and 10th eigenvectors
    eigenvectors_path = [sorted_eigvecs[:,0], sorted_eigvecs[:,1], sorted_eigvecs[:,5], sorted_eigvecs[:,9]]
    eigvectors_names = [1, 2, 5, 10]


    fig , ax = plt.subplots(2, 2, figsize=(7*2, 7*2))
    for i, vecs in enumerate(eigenvectors_path):
        #plt.subplot(1, 4, i+1)
        row = i //2
        col = i %2
        nx.draw_networkx_nodes(G, pos, ax=ax[row][col], node_color=np.round(vecs,4), node_size=20, alpha=0.8, cmap=plt.cm.jet)
        # Draw edges with curves
        for (u, v, d) in G.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        # If the nodes are neighbors, draw straight edge
            if dist <= 1:
                ax[row][col].plot([x1, x2], [y1, y2], color="gray", linewidth=1, ls='-')
            else: #otherwise, curved
                rad = 0.03  # curve radius
                arrow = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}",
                                        mutation_scale=10, lw=0.7, arrowstyle="-", color="gray", ls='--')
                ax[row][col].add_patch(arrow)
        ax[row][col].set_title(f'Eigenvector {eigvectors_names[i]}')
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])
    plt.tight_layout()
    plt.savefig('./eigvectors_graph.png')
    plt.show()


    # Comparison to theoritic results:
    L_theoritic = np.kron(L1, np.eye(n2)) + np.kron(np.eye(n1), L2)
    assert np.allclose(L, L_theoritic) 

    # Check if the eigenvectors match the theory
    # i.e  psi_i \odot psi_j is eigenvector of L with eigenvalue \lambda_i + \mu_j
    for i in range(sorted_psis.shape[1]):
        for j in range(sorted_phis.shape[1]):
            assert np.allclose(
                np.dot(L, np.kron(sorted_psis[:,i], sorted_phis[:,j])), 
                (sorted_lambdas[i] + sorted_mus[j]) * np.kron(sorted_psis[:,i], sorted_phis[:,j])
            ), f"Assertion failed for i={i}, j={j}"

if __name__ == '__main__':
    main()