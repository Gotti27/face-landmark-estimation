from torch import nn
import torch


class WingLoss(nn.Module):
    def __init__(self, width=5, curvature=0.5, method="mean"):
        super(WingLoss, self).__init__()
        self.width = width
        self.curvature = curvature
        self.C = self.width - self.width * torch.log(torch.tensor(1 + self.width / self.curvature))

        valid_methods = ['mean', 'sum']
        if method not in valid_methods:
            raise ValueError("method must either be mean or sum")
        self.method = method

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        diff = prediction - target
        diff_abs = diff.abs()
        loss = diff_abs.clone()

        smaller_mask = diff_abs < self.width
        geq_mask = diff_abs >= self.width

        loss[smaller_mask] = self.width * torch.log(1 + loss[smaller_mask] / self.curvature)
        loss[geq_mask] = loss[geq_mask] - self.C

        if self.method == 'mean':
            loss = loss.mean()
        elif self.method == 'sum':
            loss = loss.sum()
        else:
            raise ValueError("invalid method")

        return loss
