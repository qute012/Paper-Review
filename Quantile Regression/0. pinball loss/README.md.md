## Pinball Loss

Quantile regression을 위한 loss function으로 Pinball loss를 많이 쓴다.

어떻게 보면 평균 절대 오차(Mean absolute error)인 MAE에 quantile 만큼의 가중치를 부여한 loss function이라고 이해하였다. 수식은 아래와 같다.

![image-20210107012633075](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210107012633075.png)

최종적으로 나오는 값은 무조건 양수이며, quantile 값에 따라 최종 loss 값에 가중치가 부여되며, 아래와 같이 0이하의 값에서는 overforecasting되고 0이상의 값에서는 underforcasting 된다.

![image-20210107021222876](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210107021222876.png)

TF implementation

```
from tensorflow.keras.backend import mean, maximum

def quantile_loss(q, y, pred):
  err = (y-pred)
  return mean(maximum(q*err, (q-1)*err), axis=-1)
```





Pytorch implementation

```
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
```



딥러닝 기반의 모델에 pinball loss를 적용하였을 때, SGD보다는 Adam이 더욱 효과가 있었다.