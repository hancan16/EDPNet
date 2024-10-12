import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from .modules import Conv2dWithConstraint


class EEGNetEncoder(nn.Module):
    def __init__(
        self,
        samples=1000,
        channels=20,
        dropoutRate=0.2,
        kernelLength=64,
        kernelLength2=16,
        F1=8,
        D=2,
        F2=16,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            # 时间信息融合，时间编码器
            nn.Conv2d(1, F1, (1, kernelLength), stride=1, padding="same", bias=False),
            nn.BatchNorm2d(F1),
            # DepthwiseConv2D =======================
            # 空间信息融合，空间滤波器
            Conv2dWithConstraint(
                F1,
                F1 * D,
                (channels, 1),
                max_norm=1,
                stride=1,
                padding=(0, 0),
                groups=F1,
                bias=False,
            ),  # groups=F1 for depthWiseConv
            # ========================================
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropoutRate),
        )
        self.block2 = nn.Sequential(
            # SeparableConv2D =======================
            nn.Conv2d(
                F1 * D,
                F1 * D,
                (1, kernelLength2),
                stride=1,
                padding="same",
                bias=False,
                groups=F1 * D,
            ),
            nn.Conv2d(F1 * D, F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            # ========================================
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropoutRate),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class MaxNormLinear(nn.Linear):
    """Linear layer with MaxNorm constraining on weights.

    Equivalent of Keras tf.keras.Dense(..., kernel_constraint=max_norm())
    [1, 2]_. Implemented as advised in [3]_.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    out_features: int
        Size of each output sample.
    bias: bool, optional
        If set to ``False``, the layer will not learn an additive bias.
        Default: ``True``.

    References
    ----------
    .. [1] https://keras.io/api/layers/core_layers/dense/#dense-class
    .. [2] https://www.tensorflow.org/api_docs/python/tf/keras/constraints/
           MaxNorm
    .. [3] https://discuss.pytorch.org/t/how-to-correctly-implement-in-place-
           max-norm-constraint/96769
    """

    def __init__(
        self, in_features, out_features, bias=True, max_norm_val=2, eps=1e-5, **kwargs
    ):
        super().__init__(
            in_features=in_features, out_features=out_features, bias=bias, **kwargs
        )
        self._max_norm_val = max_norm_val
        self._eps = eps

    def forward(self, X):
        self._max_norm()
        return super().forward(X)

    def _max_norm(self):
        with torch.no_grad():
            norm = self.weight.norm(2, dim=0, keepdim=True).clamp(
                min=self._max_norm_val / 2
            )
            desired = torch.clamp(norm, max=self._max_norm_val)
            self.weight *= desired / (self._eps + norm)


class EEGNet(nn.Module):
    def __init__(
        self,
        n_classes=2,
        channels=20,
        samples=1000,
        dropoutRate=0.2,
        kernelLength=64,
        kernelLength2=16,
        F1=8,
        D=2,
        F2=16,
    ):
        super(EEGNet, self).__init__()

        self.encoder = EEGNetEncoder(
            samples, channels, dropoutRate, kernelLength, kernelLength2, F1, D, F2
        )

        inp = torch.ones(1, 1, channels, samples)
        out = self.encoder(inp)
        out_shape = out.cpu().data.numpy().shape
        T = out_shape[-1] * out_shape[-2] * out_shape[-3]
        # T = int(samples / 32)

        # self.classifier = nn.Linear(T, n_classes, bias=False)
        self.classifier = MaxNormLinear(T, n_classes, bias=False, max_norm_val=0.25)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(x.size()[0], -1)  # Flatten
        x = self.classifier(x)

        return x
