import torch
import torch.nn as nn
import torch.nn.functional as F


class Spatial_BCE_Loss(nn.Module):
    def __init__(self):
        super(Spatial_BCE_Loss, self).__init__()
        self.eps = 1e-8

    def forward(self, x, y, threshold_p=None, fg=None, iter=0):
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)
        x_sigmoid = torch.sigmoid(x)
        if iter % 100 < 80:
            threshold = x_sigmoid.view(b * c, h * w)
            threshold = torch.sort(threshold, descending=True)[0]
            index = fg * (h * w)
            index = index.view(-1)
            index = index.int()
            mask = torch.zeros(b*c, h*w)
            for i in range(b*c):
                mask[i][index[i]] = 1
            threshold = torch.masked_select(threshold, mask.bool())
            threshold = threshold.view(b, c)
            threshold = torch.stack([threshold for _ in range(h * w)], dim=2)
            threshold = threshold.detach()
        else:
            threshold = threshold_p.view(b, c, h*w)
            threshold = torch.mean(threshold, dim=2)
            threshold = F.sigmoid(threshold)
            threshold = torch.stack([threshold for _ in range(h * w)], dim=2)
            x_sigmoid = x_sigmoid.detach()
        threshold = threshold.clamp(min=0.0001)
        y_ = torch.stack([y for _ in range(h * w)], dim=2)
        mask_low = (x_sigmoid <= threshold) * y_
        mask_high = (x_sigmoid > threshold) * y_
        mask_low = mask_low.detach()
        mask_high = mask_high.detach()
        h_low = (-(torch.pow(x_sigmoid, 2)/torch.pow(threshold, 2)) + 2 * x_sigmoid / threshold) * mask_low
        alpha = 1/torch.pow(1-threshold, 2).clamp(min=self.eps)
        h_high = (alpha * (1 - x_sigmoid) * (1 - 2 * threshold + x_sigmoid)) * mask_high
        if iter % 100 < 80:
            piecewise_spatial_bceloss = h_high + h_low
            neg_loss = -(1 - y_) * torch.log((1 - x_sigmoid).clamp(min=self.eps))
            piecewise_spatial_bceloss = piecewise_spatial_bceloss + neg_loss
            piecewise_spatial_bceloss = torch.mean(piecewise_spatial_bceloss)
        else:
            piecewise_spatial_bceloss = torch.sum(h_low) / torch.sum(mask_low) + torch.sum(h_high) / torch.sum(mask_high)

        return piecewise_spatial_bceloss


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        self.eps = 1e-8

    def forward(self, x, y, fg):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        prob = torch.stack([xs_neg, xs_pos], dim=2)
        marginal = torch.mean(prob, dim=3)

        y_ = torch.stack([y, y], dim=2)

        bg = 1 - fg
        fb = torch.stack([bg, fg], dim=2)

        kl_loss = (y_ * marginal * torch.log((marginal / (fb + 1e-10)).clamp(min=self.eps))).sum(2)
        kl_loss = torch.sum(kl_loss) / torch.sum(y)
        return kl_loss






