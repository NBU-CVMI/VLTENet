import torch
import torch.nn as nn
import torch.nn.functional as F



class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))

        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, ind, target,reg_mask):
        pred = self._tranpose_and_gather_feat(output, ind) # 17,2
        loss = F.l1_loss(pred, target, reduction='sum')

        loss = loss / (reg_mask.sum() + 1e-4)

        return loss


class MseWight(nn.Module):
    def __init__(self):
        super(MseWight, self).__init__()
    def forward(self, pred, gt):
        criterion = nn.MSELoss(reduction='none')
        loss = criterion(pred, gt)
        ratio = torch.pow(50, gt)
        loss = torch.mul(loss, ratio)
        loss = torch.mean(loss)
        return loss


class LossAll(torch.nn.Module):

    def __init__(self):
        super(LossAll, self).__init__()

        self.L_hm = MseWight()
        self.vec = RegL1Loss()


    def forward(self, pr_decs, gt_batch):
        hm_loss = self.L_hm(pr_decs['hm'], gt_batch['hm'])
        vec_loss = self.vec(pr_decs['vec_ind'], gt_batch['ind'], gt_batch['vec_ind'], gt_batch['reg_mask'])
        loss_dec = hm_loss + vec_loss

        return loss_dec, hm_loss, vec_loss



