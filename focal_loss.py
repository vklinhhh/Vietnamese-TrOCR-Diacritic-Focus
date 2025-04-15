import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance and focusing on hard examples.
    
    Paper: "Focal Loss for Dense Object Detection" - https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        # Calculate standard cross entropy loss
        ce_loss = self.ce_loss(inputs, targets)
        
        # Calculate probabilities of the target classes
        pt = torch.exp(-ce_loss)
        
        # Apply focal weighting: (1-pt)^gamma reduces the loss for well-classified examples
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

