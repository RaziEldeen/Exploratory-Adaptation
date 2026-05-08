import torch
import torch.nn as nn
from torch.autograd import Function
import math

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


class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, output_dim=1, masks=None, learning_rule='bp', problem_type='classification', device=torch.device("cpu")):
        super(FeedforwardNN, self).__init__()
        self.device = device
        if masks is not None:
            self.masks = [mask.to(self.device) for mask in masks]
            self.n_hidden = len(masks)
        else:
            self.masks = [torch.ones(input_dim, input_dim).to(self.device)]
            self.n_hidden = 1
        self.learning_rule = learning_rule
        self.problem_type = problem_type
        self.out_dim = output_dim

        layers = []
        for i in range(self.n_hidden):
            in_dim = input_dim if i == 0 else self.masks[i].shape[1]
            out_dim = self.masks[i].shape[0]
            layer_class = FeedbackAlignmentLinear if self.learning_rule == 'fa' else MaskedLinear
            layers.append(layer_class(in_dim, out_dim, self.masks[i], self.device))
            layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

        last_hidden_dim = self.masks[-1].shape[0]
        self.output_layer = nn.Linear(last_hidden_dim, self.out_dim).to(self.device)

        if self.problem_type == 'classification':
            self.output_activation = nn.Softmax(dim=1) if self.out_dim > 1 else nn.Sigmoid()
        elif self.problem_type == 'regression':
            self.output_activation = None
        else:
            raise ValueError("Invalid problem_type. Choose either 'classification' or 'regression'.")

        self.intermediate_outputs = {}

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, nn.Tanh):
                self.intermediate_outputs[f"layer_{i // 2}"] = x.detach().clone()
        x = self.output_layer(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x

    def get_intermediate_outputs(self):
        return self.intermediate_outputs

