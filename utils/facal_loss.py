import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.05, gamma=1, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        #eps = 0.0001
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            sigmoid = nn.Sigmoid()
            BCE_loss = F.binary_cross_entropy(sigmoid(inputs), targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = - self.alpha * (1 - pt) ** self.gamma * (targets * torch.log(pt)) - \
                 (1 - self.alpha) * (1 - pt) ** self.gamma *((1 - targets) * torch.log(pt))
        print('1:', torch.mean(- self.alpha * (1 - pt) ** self.gamma * (targets * torch.log(pt))))
        print('0:', torch.mean(- (1 - self.alpha) * (1 - pt) ** self.gamma * ((1 - targets) * torch.log(pt))))
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def Facal_loss(input, target):
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + FocalLoss().forward(c[0], c[1])

        return s / (i + 1)
