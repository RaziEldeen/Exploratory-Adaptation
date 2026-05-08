from experiment_manager import run_simulation_ff, train_and_test
from data_loader import load_data

def main():
    # Synthetic dataset
    train_data, test_data = load_data(K=2, D=500, N=10, test_size=0.3, random_state=42, problem_type='classification', dataset='synthetic')
    training_errors, test_errors, best_hyperparams_str = run_simulation_ff(train_data, test_data, problem_type='classification', trials=1, epochs=5)
    print("Best hyperparameters for synthetic dataset:", best_hyperparams_str)

    # MNIST dataset
    train_data, test_data = load_data(K=10, D=784, N=60, test_size=0.3, random_state=42, problem_type='classification', dataset='mnist')
    training_errors, test_errors, best_hyperparams_str = run_simulation_ff(train_data, test_data, problem_type='classification', trials=1, epochs=5)
    print("Best hyperparameters for MNIST dataset:", best_hyperparams_str)


if __name__ == '__main__':
    main()