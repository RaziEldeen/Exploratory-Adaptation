from train_test_utils import *
from feedforward_nn import FeedforwardNN
from data_loader import load_data
from torch.utils.data import DataLoader
import numpy as np
from scipy import stats
import torch.optim as optim
import torch.nn as nn




def get_input_dim(data):
    if hasattr(data, "tensors"):
        return data.tensors[0][0].size(0)
    else:
        sample_input, _ = data[0]
        return sample_input.size(0)
    
def get_output_dim(data):
    if hasattr(data, "classes"):
        return len(data.classes)
    elif hasattr(data, "targets"):
        return len(torch.unique(data.targets))
    elif hasattr(data, "tensors"):
        return data.tensors[1].unique().size(0)
    else:
        return None

def run_simulation_ff(train_data, test_data, masks=None, problem_type='classification', trials=10, epochs=20, seed = 42, device=None):
    device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if masks is not None:
        D = masks[0].shape[1]
    else:
        D = get_input_dim(train_data)

    if problem_type == 'classification':
        out_dim = get_output_dim(train_data)
    else:
        out_dim = 1

    print(f"in_dim = {D}, out_dim = {out_dim}")
    #random learning rates
    np.random.seed(seed)  # Set a seed for reproducibility
    num_random_lr = 5
    min_lr = 1e-5
    max_lr = 1e-1

    random_lrs = np.random.uniform(min_lr, max_lr, num_random_lr)

    # Define hyperparameter grid
    hyperparameter_grid = [
        {'learning_rule': lr, 'lr': learning_rate, 'batch_size': bs, 'n_hidden': nh}
        for lr in ['fa', 'bp']
        for learning_rate in random_lrs
        for bs in [128]
        for nh in [len(masks)]  # Add more values for n_hidden as needed
    ]

    best_hyperparams = None
    training_errors = {}
    test_errors = {}
    best_error = np.inf
    # Grid search over hyperparameters
    for trial in range(trials):
        for hyperparams in hyperparameter_grid:
            hyperparams['trial'] = trial

            print(f"Training with {hyperparams}")

            model = FeedforwardNN(D,output_dim=out_dim, masks=masks, 
                                  learning_rule=hyperparams['learning_rule'], problem_type=problem_type,
                                  device = device)
            optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])

            # Change batch size in dataloaders
            train_loader = DataLoader(train_data, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=5)
            test_loader = DataLoader(test_data, batch_size=hyperparams['batch_size'], num_workers=5)

            # Train and test the model
            training_error, test_error = train_and_test(model, train_loader, test_loader, optimizer
                                                        , hyperparams['learning_rule'], problem_type=problem_type
                                                        , epochs=epochs, device=device)
            best_test_error = test_error[-1]
            if best_test_error < best_error:
                best_error = best_test_error
                best_hyperparams = hyperparams
                best_hyperparams_str = f"full_{hyperparams['learning_rule']}_lr{hyperparams['lr']}_batch{hyperparams['batch_size']}_n_hidden{hyperparams['n_hidden']}_trial{hyperparams['trial']}"

            # Save training and test errors
            hyperparams_str = f"full_{hyperparams['learning_rule']}_lr{hyperparams['lr']}_batch{hyperparams['batch_size']}_n_hidden{hyperparams['n_hidden']}_trial{hyperparams['trial']}"
            training_errors[hyperparams_str] = training_error
            test_errors[hyperparams_str] = test_error

    return training_errors, test_errors, best_hyperparams_str



def run_simulation_ff_multi_masks(train_data, test_data, masks, problem_type='classification', trials=10, epochs=20, seed = 42, device = None):
    mask_results = {}
    #masks should be a list of lists
    for mask_idx, mask in enumerate(masks):
        print(f"Running experiments for mask {mask_idx + 1}")
        print(len(mask))
        training_errors, test_errors, best_hyperparams_str = run_simulation_ff(train_data, test_data, mask, problem_type = problem_type, trials=trials, epochs=epochs, seed = seed, device=device)
        
        mask_results[mask_idx] = {
            'training_errors': training_errors,
            'test_errors': test_errors,
            'best_hyperparams': best_hyperparams_str
        }

    return mask_results


def compare_best_masks(training_errors1, training_errors2, alpha=0.05):
    """
    Compares the best training errors of two experiments (with different masks) using a two-sample t-test.

    Args:
        training_errors1 (dict): Dictionary of training errors for the first mask.
        training_errors2 (dict): Dictionary of training errors for the second mask.
        alpha (float): Significance level for the t-test.

    Returns:
        A dictionary with the results of the t-test for the best training errors of each pair of experiments (one from each mask).
    """
    results = {}

    # Get the best training error for each experiment
    best_errors1 = [min(errors) for errors in training_errors1.values()]
    best_errors2 = [min(errors) for errors in training_errors2.values()]

    # Calculate the t-statistic and p-value for the two-sample t-test
    t_stat, p_value = stats.ttest_ind(best_errors1, best_errors2)

    # Check if the p-value is less than the significance level (alpha)
    significant = p_value < alpha

    results['best_training_errors'] = {'t_stat': t_stat, 'p_value': p_value, 'significant': significant}

    return results