import torch
import torch.nn

class PinballLoss(nn.Module):
    def __init__(self, quantile=0.10, reduction='mean'):
        super(PinballLoss, self).__init__()
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction
        
    def forward(self, output, target):
        errors = target - output
        if self.reduction=='mean':
            return torch.mean(torch.max((self.quantile-1) * errors, self.quantile * errors))
        elif self.reduction=='sum':
            return torch.sum(torch.max((self.quantile-1) * errors, self.quantile * errors))

from tensorflow.keras.backend import mean, maximum

def quantile_loss(q, y, pred):
  err = (y-pred)
  return mean(maximum(q*err, (q-1)*err), axis=-1)