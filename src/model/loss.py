## author: xin luo
## created: 2022.4.3
# Developer: Jin Ikeda
# Last modified Dec 5, 2024
## des: loss function

import torch
import torch.nn as nn
import torch.nn.functional as F

# Focal Loss fof binary classification
class FocalLoss_binary(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss_binary, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class FocalLoss_multi(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduction='mean'):
        """
        Multi-Class Focal Loss

        Parameters:
            alpha (float): Weighting factor for the classes.
            gamma (float): Focusing parameter to adjust the impact of easy vs. hard examples.
            logits (bool): If True, the input is raw logits, and softmax will be applied.
            reduction (str): Specifies the reduction to apply to the output ('mean', 'sum', or 'none').
        """
        super(FocalLoss_multi, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for the Focal Loss computation.

        Parameters:
            inputs (torch.Tensor): Predictions (logits or probabilities) of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth class indices of shape (batch_size).

        Returns:
            torch.Tensor: Computed focal loss.
        """
        if self.logits:
            # Apply softmax to logits to get probabilities
            inputs = F.softmax(inputs, dim=1)

        # Convert class indices to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        # Compute the cross-entropy term
        cross_entropy_loss = -targets_one_hot * torch.log(inputs + 1e-9)

        # Compute pt (probability of the true class)
        pt = (inputs * targets_one_hot).sum(dim=1)

        # Compute the focal loss
        focal_term = (1 - pt) ** self.gamma
        focal_loss = self.alpha * focal_term * cross_entropy_loss.sum(dim=1)

        # Apply reduction (mean, sum, or none)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss