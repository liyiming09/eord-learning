from packaging import version
import torch
from torch import nn
import math
from .sinkhorn import OT

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class CoNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.l2_norm = Normalize(2)

    def forward(self, feat_q, feat_k, i):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        # Therefore, we will include the negatives from the entire minibatch.
        # if self.opt.nce_includes_all_negatives_from_minibatch:
        #     batch_dim_for_bmm = 1
        # else:
        if self.opt.divco:
            batch_dim_for_bmm = self.opt.batchSize // len(self.opt.gpu_ids) * (self.opt.num_negative + 2)
        else:
            batch_dim_for_bmm = self.opt.batchSize // len(self.opt.gpu_ids)

        # print('feat_q', feat_q.shape)
        ot_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        ot_k = feat_k.view(batch_dim_for_bmm, -1, dim).detach()
        # pos_weight = torch.bmm(feat_c.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        # pos_weight = pos_weight.view(batchSize, 1)
        # print(i,'ot:',  ot_q.shape,ot_k.shape)
        # ot_q = torch.randn(1, 256, 10)
        # ot_k = torch.randn(1, 256, 10)
        # ot_q = torch.tensor([[[0.6, 0.8],[0.3, 0.95]]])
        # ot_k = torch.tensor([[[0.3, 0.95],[0.6, 0.8]]])

        f = OT(ot_q, ot_k, eps=1.0, max_iter=50, cost_type = self.opt.cost_type)
        f = f.permute(0, 2, 1) * self.opt.ot_weight + 1e-8
        f_max = torch.max(f, -1)[0].view(batchSize, 1)
        # f_tmp = f[0, 0, 1:]
        # print('*****', f_tmp.min(), f_tmp.max()) # tensor(0.3630) tensor(0.6370)
        # print('*****', f[0, 10, 10], f[0, 9:12, 9:12])
        # 1/0

        feat_k = feat_k.detach()
        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        # print(i,'pos:',l_pos.shape, 'f_max',f_max.shape)
        if i == 4:
            l_pos = l_pos.view(batchSize, 1) + torch.log(f_max) * 0.07
        else:
            l_pos = l_pos.view(batchSize, 1)

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
        if i == 4:
            l_neg_curbatch = l_neg_curbatch + torch.log(f) * 0.07
        # print(i, 'feat_q.feat_k', feat_q.shape, feat_k.shape)
        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)
        # print(i,'neg:',l_neg.shape, 'diagonal',diagonal.shape)

        out = torch.cat((l_pos, l_neg), dim=1) / 0.07
        # print(i,'out:',out.shape)
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device)).mean()
        # print(loss.mean().shape)
        # print('*****cross_entro', loss.mean(), loss2)
        # 1/0

        return loss

class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)
        # print('pos:',l_pos.shape)
        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        # if self.opt.nce_includes_all_negatives_from_minibatch:
        #     # reshape features as if they are all negatives of minibatch of size 1.
        #     batch_dim_for_bmm = 1
        # else:
        if self.opt.divco:
            batch_dim_for_bmm = self.opt.batchSize // len(self.opt.gpu_ids) * (self.opt.num_negative + 2)
        else:
            batch_dim_for_bmm = self.opt.batchSize // len(self.opt.gpu_ids)

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / 0.07

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device)).mean()

        return loss