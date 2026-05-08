import numpy as np
import matplotlib.pyplot as plt



def loss(A,b,x):
    return 0.5 * np.linalg.norm(np.dot(A, x) - b) ** 2
def dloss(A,b,x):
    return np.dot(A.T, (np.dot(A, x) - b))

#  Gradient descent for smooth convex optimization
def gradient_descent(A, b, x_init, n_iterations, beta):
    x = x_init
    losses = [loss(A,b,x)]
    xs = [x]
    for t in range(2,n_iterations+1):
        grad = dloss(A,b,x)
        x = x - (1.0/beta) * grad
        xs.append(x)
        losses.append(loss(A,b,x))
    return xs, losses


# Accelerated gradient method - O(1/t^2) rate
def accelerated_gradient_descent(A, b, x_init, n_iterations, beta):
    eta = 1
    x = y = x_init
    losses = [loss(A,b,x)]
    ys = [y]
    for t in range(2,n_iterations+1):
        z = (1-eta)*y + eta*x
        gradient = dloss(A,b,z)
        x = x - (1.0/(eta*beta)) * gradient #x_{t+1}
        y = (1-eta)*y + eta*x
        ys.append(y)
        losses.append(loss(A,b,y))
        eta = (-eta**2 + np.sqrt(eta**4 + 4*eta**2))/2.0
        ## for theoritical comparison
        
    return ys, losses


# Subgradient method for non-smooth optimization
def subgradient_method(A, b, x_init, n_iterations, D, G):
    x = x_init
    losses = [loss(A,b,x)]
    xs = [x]
    for t in range(2, n_iterations + 1):
        grad = dloss(A,b,x)
        lr = D / (G * np.sqrt(t))
        x = x - lr * grad
        xs.append(x)
        losses.append(loss(A,b,x))
    return xs, losses


# Generate data

def generate_data(d, sigma_max, sigma_min, noise_scale):
    # Generate a random matrix with fixed singular values
    U, _ = np.linalg.qr(np.random.rand(d, d))
    V, _ = np.linalg.qr(np.random.rand(d, d))
    singular_values = np.linspace(sigma_max, sigma_min, d)
    A = U @ np.diag(singular_values) @ V.T

    # Generate a solution x_star
    x_star = 0.05* np.random.randn(d, 1)

    # Generate the right-hand side b
    b = A @ x_star + noise_scale * np.random.rand(d, 1)

    #x_0
    x_init = np.zeros(d)
    # Compute the beta, D, G
    beta = sigma_max**2
    D = np.linalg.norm(x_init - x_star)
    G = sigma_max * np.linalg.norm(np.dot(A, x_init) - b) 
    
    return A, b, beta, D, G, x_init, x_star


# Experiment
def experiment(d, n_iterations, n_experiments, noise_scale, sigma_max, sigma_min):
    gd_losses = []
    agd_losses = []
    sg_losses = []
    for _ in range(n_experiments):
        A, b, beta, D, G, x_init, x_star = generate_data(d, sigma_max, sigma_min, noise_scale)
        _, gd_loss = gradient_descent(A, b, x_init, n_iterations, beta=beta)
        _, agd_loss = accelerated_gradient_descent(A, b, x_init, n_iterations, beta=beta)
        _, sg_loss = subgradient_method(A, b, x_init, n_iterations, D, G)
        gd_losses.append(gd_loss)
        agd_losses.append(agd_loss)
        sg_losses.append(sg_loss)
    gd_losses = np.mean(gd_losses, axis=0)
    agd_losses = np.mean(agd_losses, axis=0)
    sg_losses = np.mean(sg_losses, axis=0)
    return gd_losses, agd_losses, sg_losses



def main():
    # Constants
    d = 5
    n_iterations = 50
    n_experiments = 1
    noise_scale = 0.5
    sigma_max = 3.0
    sigma_min = 0.5


    # # Run experiments for both cases
    gd_losses_pd, agd_losses_pd, sg_losses_pd = experiment(d, n_iterations, n_experiments, noise_scale,
                                                         sigma_max=sigma_max, sigma_min=sigma_min)
    gd_losses_npd, agd_losses_npd, sg_losses_npd = experiment(d, n_iterations, n_experiments, noise_scale,
                                                             sigma_max=sigma_max,
                                                             sigma_min=0)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)
    t = np.arange(1, n_iterations+1)
    # Plot results for positive-definite case
    axs[0].plot(t,gd_losses_pd, label='Smooth Gradient Descent')
    axs[0].plot(t, agd_losses_pd, label='Accelerated Gradient Descent')
    axs[0].plot(t, sg_losses_pd, label='Gradient Descent')
    axs[0].legend()
    axs[0].grid()
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss')
    #axs[0].set_yscale('log')
    axs[0].set_title('Positive-definite case')
    #axs[0].set_xlim((1, n_iterations))

    # Plot results for non-positive-definite case
    axs[1].plot(t,gd_losses_npd, label='Smooth Gradient Descent')
    axs[1].plot(t, agd_losses_npd, label='Accelerated Gradient Descent')
    axs[1].plot(t, sg_losses_npd, label='Gradient Descent')
    axs[1].legend()
    axs[1].grid()
    axs[1].set_xlabel('Iteration')
    axs[1].set_title('Non-positive-definite case')

    plt.tight_layout()
#    plt.savefig('./results.jpeg')  # save before showing
    plt.show()

## comparison to theoritical results
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)
    A, b, beta, D, G, x_init, x_star =  generate_data(100, sigma_max, 0, 0)
    t = np.arange(4, n_iterations+1)
    xs, losses_acc = accelerated_gradient_descent(A,b, x_init, n_iterations, beta)
    xs, losses = gradient_descent(A,b, x_init, n_iterations, beta)

    theoritical_acc = loss(A,b, x_star) + 32*beta*np.linalg.norm(x_init - x_star)**2/(9*(t-1)**2)
    theoritical = loss(A,b, x_star) + 4*beta*np.linalg.norm(x_init - x_star)**2/(t-1)

    axs[0].plot(t, losses[3:],'.', t, theoritical)
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Iteration')
    axs[0].legend(['experiemnet','theory'])
    axs[0].set_title('Smooth Gradient Descent')
    axs[1].plot(t, losses_acc[3:],'.', t, theoritical_acc)
    axs[1].set_ylabel('Loss')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Iteration')
    axs[1].legend(['experiemnet','theory'])
    axs[1].set_title('Accelerated Gradient Descent')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()