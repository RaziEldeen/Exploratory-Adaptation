import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_vectors(dim, n_samples):
    return np.random.normal(0, 1, (n_samples, dim))

def calculate_distances(points):
    n_samples = points.shape[0]
    distances = []
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            distances.append(np.linalg.norm(points[i] - points[j]))
    return distances

def plot_distances_histogram(distances, bins=50):
    plt.hist(distances, bins=bins, density=True)
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.title("Pairwise Distances Between Gaussian Random Vectors")
    plt.show()

def plot_radius(points):
    distances = np.sqrt((points **2).sum(1))
    plt.hist(distances, bins = 50)
    plt.show()

def main():
    import numpy as np

    # Create a 3x3x3 tensor
    n = 3
    tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])

    # Stack the tensor as a vector
    v = tensor.ravel(order='F')


    I_n = np.eye(n)
    P_n = np.fliplr(I_n)
    prod = np.kron(I_n,P_n)  
    R_90 = np.kron(np.eye(3),P_n)
    # Reshape the rotated vector back into a tensor
    v_rotated = R_90 @ v
    rotated_tensor = np.reshape(v_rotated, (n, n, 3))
    import matplotlib.pyplot as plt

# ... (use the code from the previous answer to create the original and rotated tensors) ...

# Normalize the tensor values to the range [0, 1]
    tensor_normalized = tensor / np.max(tensor)
    rotated_tensor_normalized = rotated_tensor / np.max(rotated_tensor)

    # Plot the original and rotated tensors as images
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))

    ax1.imshow(tensor_normalized)
    ax1.set_title("Original tensor")
    ax1.axis("off")

    ax2.imshow(rotated_tensor_normalized)
    ax2.set_title("Rotated tensor (90 degrees clockwise)")
    ax2.axis("off")

    tensor_rotated_90_clockwise = np.rot90(tensor_normalized, k=-1, axes=(0, 1))
    ax3.imshow(tensor_rotated_90_clockwise)
    ax3.set_title("Rotated tensor (90 degrees clockwise, np)")
    ax3.axis("off")

    plt.show()



if __name__ == "__main__":
    main()
