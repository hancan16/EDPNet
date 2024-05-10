import torch
import torch.nn as nn


class PrototypeLoss(nn.Module):

    def forward(self, features, proxy, labels):

        label_prototypes = torch.index_select(proxy, dim=0, index=labels)

        pl = huber_loss(features, label_prototypes, sigma=1)
        pl_loss = torch.mean(pl)

        return pl_loss


def huber_loss(input, target, sigma=1):
    beta = 1.0 / (sigma**2)
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff**2 / beta, diff - 0.5 * beta)

    return torch.sum(loss, dim=1)
