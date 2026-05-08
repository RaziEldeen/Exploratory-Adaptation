from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST as TorchMNIST
from torchvision.transforms import Compose, Normalize, ToTensor, Lambda
from PIL import Image
from sklearn.datasets import load_iris, load_diabetes

class MNISTDataset(TorchMNIST):
    def __init__(self, *args, **kwargs):
        super(MNISTDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
        
        # Flatten the image
        img = torch.flatten(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def load_data(K=2, D=500, N=10000, test_size=0.3, random_state=42, problem_type='classification', dataset='synthetic'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if problem_type == 'classification':
        if dataset == 'iris':
            data = load_iris()
            X, y = data.data, data.target
        elif dataset == 'mnist':
            # Define the necessary transformations
            mnist_transforms = Compose([
                ToTensor(),
                Normalize((0.1307,), (0.3081,))
            ])
            
            mnist_train = MNISTDataset(root='./data', train=True, download=True, transform=mnist_transforms)
            mnist_test = MNISTDataset(root='./data', train=False, download=True, transform=mnist_transforms)
            return mnist_train, mnist_test
        else:  # synthetic dataset
            X, y = make_classification(n_samples=N, n_classes=K, n_features=D, n_informative=D, n_redundant=0, random_state=1, class_sep=1)
        y_dtype = torch.long
    elif problem_type == 'regression':
        if dataset == 'boston':
            data = load_diabetes()
            X, y = data.data, data.target
        else:  # synthetic dataset
            X, y = make_regression(n_samples=N, n_features=D, n_informative=D, random_state=1)
        y_dtype = torch.float32
    else:
        raise ValueError("Invalid problem_type. Choose either 'classification' or 'regression'.")


    
    if dataset != 'mnist':
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=y_dtype)
        y_test = torch.tensor(y_test, dtype=y_dtype)
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)

    return train_data, test_data


# def load_data(K=2, D=500, N=10000, test_size=0.3, random_state=42, problem_type='classification'):
#     # Generate synthetic dataset
#     if problem_type == 'classification':
#         X, y = make_classification(n_samples=N, n_classes=K, n_features=D, n_informative=D, n_redundant=0, random_state=1, class_sep=1)
#         y_dtype = torch.long
#     elif problem_type == 'regression':
#         X, y = make_regression(n_samples=N, n_features=D, n_informative=D, random_state=1)
#         y_dtype = torch.float32
#     else:
#         raise ValueError("Invalid problem_type. Choose either 'classification' or 'regression'.")

#     X = StandardScaler().fit_transform(X)

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

#     # Convert to PyTorch tensors and create dataloaders
#     train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=y_dtype))
#     test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=y_dtype))

#     return train_data, test_data

