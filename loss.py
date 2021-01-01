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

if __name__ == "__main__":
    label = torch.rand(8,1, 256, 256).long()
    pre = torch.rand(8, 1, 256, 256).long()
    diceloss = DiceLoss()
    celloss = CELDice()
    print(diceloss(pre, label), celloss(pre, label))
