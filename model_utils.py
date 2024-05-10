import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LightweightConv1d(nn.Module):

    def __init__(
        self,
        in_channels,
        num_heads=1,
        depth_multiplier=1,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
        weight_softmax=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(
            torch.Tensor(num_heads * depth_multiplier, 1, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_heads * depth_multiplier))
        else:
            self.bias = None

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, inp):
        B, C, T = inp.size()
        H = self.num_heads

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        # input = input.view(-1, H, T)
        inp = rearrange(inp, "b (h c) t ->(b c) h t", h=H)
        if self.bias is None:
            output = F.conv1d(
                inp,
                weight,
                stride=self.stride,
                padding=self.padding,
                groups=self.num_heads,
            )
        else:
            output = F.conv1d(
                inp,
                weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                groups=self.num_heads,
            )
        output = rearrange(output, "(b c) h t ->b (h c) t", b=B)

        return output


class VarMaxPool1D(nn.Module):
    def __init__(self, T, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        self.padding = padding

    def forward(self, x):
        mean_of_squares = F.avg_pool1d(
            x**2, self.kernel_size, self.stride, self.padding
        )
        # Compute the square of the mean (E[x])^2
        square_of_mean = (
            F.avg_pool1d(x, self.kernel_size, self.stride, self.padding) ** 2
        )

        # Compute the variance: Var[X] = E[X^2] - (E[X])^2
        variance = mean_of_squares - square_of_mean
        # out = self.time_agg(variance)
        out = F.avg_pool1d(variance, variance.shape[-1])

        return out


class VarPool1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Calculate the size of the result tensor after pooling

        # Compute the mean of the squares (E[x^2])
        mean_of_squares = F.avg_pool1d(
            x**2, self.kernel_size, self.stride, self.padding
        )

        # Compute the square of the mean (E[x])^2
        square_of_mean = (
            F.avg_pool1d(x, self.kernel_size, self.stride, self.padding) ** 2
        )

        # Compute the variance: Var[X] = E[X^2] - (E[X])^2
        variance = mean_of_squares - square_of_mean

        return variance


class SSA(nn.Module):

    def __init__(self, T, num_channels, epsilon=1e-5, mode="var", after_relu=False):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

        self.GP = VarMaxPool1D(T, 250)

    def forward(self, x):
        B, C, T = x.shape

        if self.mode == "l2":
            embedding = (x.pow(2).sum((2), keepdim=True) + self.epsilon).pow(0.5)
            norm = self.gamma / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)

        elif self.mode == "l1":
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2), keepdim=True)
            norm = self.gamma / (
                torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon
            )

        elif self.mode == "var":

            embedding = (self.GP(x) + self.epsilon).pow(0.5) * self.alpha
            norm = (self.gamma) / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)

        gate = 1 + torch.tanh(embedding * norm + self.beta)

        return x * gate, gate


class Mixer1D(nn.Module):
    def __init__(self, dim, kernel_sizes=[50, 100, 250]):
        super().__init__()
        self.var_layers = nn.ModuleList()
        self.L = len(kernel_sizes)
        for k in kernel_sizes:
            self.var_layers.append(
                nn.Sequential(
                    VarPool1D(kernel_size=k, stride=int(k / 2)),
                    nn.Flatten(start_dim=1),
                )
            )

    def forward(self, x):
        B, d, L = x.shape
        x_split = torch.split(x, d // self.L, dim=1)
        out = []
        for i in range(len(x_split)):
            x = self.var_layers[i](x_split[i])
            out.append(x)
        y = torch.concat(out, dim=1)
        return y
