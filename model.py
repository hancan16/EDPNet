import math

import torch
from torch import nn

from model_utils import SSA, LightweightConv1d, Mixer1D


class Efficient_Encoder(nn.Module):

    def __init__(
        self,
        samples,
        chans,
        F1=16,
        F2=36,
        time_kernel1=75,
        pool_kernels=[50, 100, 250],
    ):
        super().__init__()

        self.time_conv = LightweightConv1d(
            in_channels=chans,
            num_heads=1,
            depth_multiplier=F1,
            kernel_size=time_kernel1,
            stride=1,
            padding="same",
            bias=True,
            weight_softmax=False,
        )
        self.ssa = SSA(samples, chans * F1)

        self.chanConv = nn.Sequential(
            nn.Conv1d(
                chans * F1,
                F2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm1d(F2),
            nn.ELU(),
        )

        self.mixer = Mixer1D(dim=F2, kernel_sizes=pool_kernels)

    def forward(self, x):

        x = self.time_conv(x)
        x, _ = self.ssa(x)
        x_chan = self.chanConv(x)

        feature = self.mixer(x_chan)

        return feature


class EDPNet(nn.Module):

    def __init__(
        self,
        chans,
        samples,
        num_classes=4,
        F1=9,
        F2=48,
        time_kernel1=75,
        pool_kernels=[50, 100, 200],
    ):
        super().__init__()
        self.encoder = Efficient_Encoder(
            samples=samples,
            chans=chans,
            F1=F1,
            F2=F2,
            time_kernel1=time_kernel1,
            pool_kernels=pool_kernels,
        )
        self.features = None

        x = torch.ones((1, chans, samples))
        out = self.encoder(x)
        feat_dim = out.shape[-1]

        # *Inter-class Separation Prototype(ISP)
        self.isp = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        # *Intra-class Compactness(ICP)
        self.icp = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        nn.init.kaiming_normal_(self.isp)

    def get_features(self):
        if self.features is not None:
            return self.features
        else:
            raise RuntimeError("No features available. Run forward() first.")

    def forward(self, x):

        features = self.encoder(x)
        self.features = features
        self.isp.data = torch.renorm(self.isp.data, p=2, dim=0, maxnorm=1)
        logits = torch.einsum("bd,cd->bc", features, self.isp)

        return logits


if __name__ == "__main__":
    # a simple test
    model = EDPNet(chans=22, samples=1000, num_classes=4)
    inp = torch.rand(1, 22, 1000)
    out = model(inp)
    print(out.shape)
