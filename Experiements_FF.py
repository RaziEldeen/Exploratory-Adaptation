import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification,make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Custom masked linear layer
class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.linear.weight.data *= self.mask
        return self.linear(x)

# Feedback Alignment Layer
class FeedbackAlignmentLinear(nn.Module):
    def __init__(self, in_features, out_features, mask):
        super(FeedbackAlignmentLinear, self).__init__()
        self.linear = MaskedLinear(in_features, out_features, mask)
        self.random_matrix = torch.randn(out_features, in_features)
        self.register_buffer('B', self.random_matrix)

    def forward(self, x):
        return self.linear(x)

    def backward(self, grad_output):
        grad_input = grad_output @ self.B
        return grad_input

# Neural network architecture
class FeedforwardNN(nn.Module):
    def __init__(self, mask, learning_rule='bp'):
        super(FeedforwardNN, self).__init__()
        self.mask = mask
        self.learning_rule = learning_rule
        self.hidden_dimension = mask.shape[0]

        if self.learning_rule == 'fa':
            self.layer1 = FeedbackAlignmentLinear(self.hidden_dimension, self.hidden_dimension, self.mask)
        else:
            self.layer1 = MaskedLinear(self.hidden_dimension, self.hidden_dimension, self.mask) # Adjust the mask as needed

        self.layer2 = nn.Tanh()
        self.layer3 = nn.Linear(D, K)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

