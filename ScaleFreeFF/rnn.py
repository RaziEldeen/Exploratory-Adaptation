import torch
import torch.nn as nn
from torch.autograd import Function
import math

# Reusing MaskedLinearFunction, MaskedLinear, FeedbackAlignmentFunction, and FeedbackAlignmentLinear classes from the given code
class MaskedLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, mask):
        ctx.save_for_backward(input, weight, bias, mask)
        weight.data *= mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input) * mask
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, device = None):
        super(MaskedLinear, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features)).to(self.device)
        self.bias = nn.Parameter(torch.Tensor(out_features)).to(self.device)
        self.mask = mask.to(self.device)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return MaskedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class FeedbackAlignmentFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, mask, B):
        ctx.save_for_backward(input, weight, bias, mask, B)
        weight.data *= mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask, B = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(B)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input) * mask
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None



class FeedbackAlignmentLinear(MaskedLinear):
    def __init__(self, in_features, out_features, mask, device = None):
        super(FeedbackAlignmentLinear, self).__init__(in_features, out_features, mask, device)
        self.B = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False).to(device)

    def forward(self, input):
        return FeedbackAlignmentFunction.apply(input, self.weight, self.bias, self.mask, self.B)

class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mask=None, n_layers=1, learning_rule='bp', problem_type='classification', device=None):
        super(SimpleRNN, self).__init__()
        if device is None:
            self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if mask is not None:
            assert input_dim == mask.shape[0], "'mask' first dimension should be the same as 'input_dim' dimension"
            self.mask = mask.to(self.device)
        else:
            self.mask = torch.ones(input_dim, input_dim).to(self.device)
        self.learning_rule = learning_rule
        self.hidden_dimension = hidden_dim
        self.problem_type = problem_type
        self.n_layers = n_layers
        self.out_dim = output_dim

        layer_class = FeedbackAlignmentLinear if self.learning_rule == 'fa' else MaskedLinear

        input_hidden_layer = layer_class(input_dim, hidden_dim, self.mask, self.device)
        hidden_hidden_layer = layer_class(hidden_dim, hidden_dim, self.mask, self.device)

        self.rnn_cell = nn.RNNCell(input_dim, hidden_dim, nonlinearity='tanh')
        self.rnn_cell.weight_ih = input_hidden_layer.weight
        self.rnn_cell.weight_hh = hidden_hidden_layer.weight
        self.rnn_cell.bias_ih = input_hidden_layer.bias
        self.rnn_cell.bias_hh = hidden_hidden_layer.bias

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        if self.problem_type == 'classification':
            self.output_activation = nn.Softmax(dim=1) if self.out_dim > 1 else nn.Sigmoid()
        elif self.problem_type == 'regression':
            self.output_activation = None
        else:
            raise ValueError("Invalid problem_type. Choose either 'classification' or 'regression'.")

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        hidden = self.init_hidden(batch_size)

        outputs = []
        for i in range(seq_len):
            hidden = self.rnn_cell(x[:, i], hidden)
            outputs.append(hidden)

        outputs = torch.stack(outputs, dim=1)
        x = self.output_layer(outputs[:, -1, :])
        if self.output_activation:
            x = self.output_activation(x)
        return x

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dimension, device=self.device)
