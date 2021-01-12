import torch.nn as nn
import torch
class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

class CELDice:
    def __init__(self, dice_weight=0,num_classes=1):
        self.nll_loss = nn.NLLLoss()
        self.jaccard_weight = dice_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
       loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)
       if self.jaccard_weight:
           eps = 1e-15
           for cls in range(self.num_classes):
               jaccard_target = (targets == cls).float()
               jaccard_output = outputs[:, cls].exp()
               intersection = (jaccard_output * jaccard_target).sum()
               union = jaccard_output.sum() + jaccard_target.sum()
               loss -= torch.log((2*intersection + eps) / (union + eps)) * self.jaccard_weight
       return loss
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

if __name__ == "__main__":
    label = torch.rand(8,1, 256, 256)
    pre = torch.rand(8, 1, 256, 256)
    diceloss = BinaryDiceLoss()
    # celloss = CELDice()
    print(diceloss(pre, label))
