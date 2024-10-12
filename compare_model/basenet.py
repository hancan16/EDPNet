import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor


class BaseNet(nn.Module):
    def __init__(
        self,
        chans,
        samples,
        num_classes,
        F1=40,
        time_kernel1=25,
        pool_kernel1=75,
        pool_stride=15,
        F2=16,
        time_kernel2=15,
        pool_kernel2=8,
    ):
        # self.patch_size = patch_size
        #
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, time_kernel1), (1, 1)),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1, (chans, 1), (1, 1), groups=F1),
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.AvgPool2d((1, pool_kernel1), (1, pool_stride)),
            nn.Dropout(0.5),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F2, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.Conv2d(F2, F2, (1, time_kernel2), stride=(1, 1), groups=F2),
            nn.Conv2d(F2, F2, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool_kernel2)),
            nn.Dropout(0.5),
        )

        inp = torch.ones((1, 1, chans, samples))
        out = self.block1(inp)
        out = self.block2(out)
        out_shape = out.size()
        T = out_shape[-1] * out_shape[-2] * out_shape[-3]

        self.classifier = nn.Linear(T, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)

        feature = torch.flatten(x, 1)
        y = self.classifier(feature)

        return y
