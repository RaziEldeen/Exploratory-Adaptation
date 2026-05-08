# test_nn.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from feedforward_nn_test import FeedforwardNN  # Replace with your module name that contains the provided code
from mask_generation import generate_random_directed_adjacency

def print_device_info(model):
    for name, param in model.named_parameters():
        print("{}: {}".format(name, param.device))

def create_dummy_data(n_samples, input_dim):
    X = np.random.rand(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, 2, size=(n_samples, 1)).astype(np.float32)
    return X, y

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # Create dummy data
    input_dim = 20
    n_samples = 1000
    X, y = create_dummy_data(n_samples, input_dim)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dataloader = DataLoader(dataset, batch_size=32)

    # Create FeedforwardNN model
    model = FeedforwardNN(input_dim, output_dim=1, n_hidden=1, learning_rule='bp', problem_type='classification', device=device).to(device)

    # Print device info of the tensors
    print_device_info(model)

    # Train the model for one epoch
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def test(m,n,seed=1):
    size = max(m,n)
    in_dist = out_dist = 'sf'
    gamma, a = 2.4, 10
    return generate_random_directed_adjacency(size, in_dist, out_dist, gamma=gamma, a=a, seed=seed)[:m,:n]
if __name__ == '__main__':
    from data_loader import load_data
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
#    from experiment_manager import run_simulation_ff
    from mask_generation import generate_random_directed_adjacency, reduce_mask
    train_data, test_data = load_data(dataset='mnist')
    train_loader = DataLoader(train_data, batch_size=1028) 
    in_dist = out_dist = 'sf'
    gamma, a = 2.4, 10
    mask = generate_random_directed_adjacency(784, in_dist, out_dist, gamma=gamma, a=a, seed=1)
    masks = [reduce_mask(mask, 100,784)]
    masks = [torch.ones(1000,784), torch.ones(100, 1000)]
#    res = run_simulation_ff(train_data, test_data, masks, trials=1, epochs=1)
#    print(res)
    from feedforward_nn_test import FeedforwardNN
    import torch.nn as nn
    model = FeedforwardNN(784, masks=masks)
    for x,y in train_loader:
        my_ = model(x)
        break

    intermediate_outputs = model.get_intermediate_outputs()
    def plot_output_distributions(intermediate_outputs):
        legends = []
        for i, (key, value) in enumerate(intermediate_outputs.items()):
            t = value
            print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, key, t.mean(), t.std(), (t.abs() > 0.97).float().mean() * 100))
            hy, hx = torch.histogram(t, bins=50, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer {i} ({key})')
        plt.legend(legends)
        plt.title('Activation Distribution')
        plt.xlabel("Activation Value")
        plt.ylabel("Density")
        plt.show()

    plot_output_distributions(intermediate_outputs)
