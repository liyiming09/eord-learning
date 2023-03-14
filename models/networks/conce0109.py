from packaging import version
import torch
from torch import nn
import math
from .sinkhorn import OT, intervOT

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
        if i == 3:
            l_pos = l_pos.view(batchSize, 1) + torch.log(f_max) * 0.07
        else:
            l_pos = l_pos.view(batchSize, 1)

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
        if i == 3:
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

class MaskCoNCELoss(nn.Module):
    def __init__(self, opt, OT = True):
        super().__init__()
        #除对应位置之外，再根据干预区域mask（比较合理的设置，三个map的并集），重新划分正负例
        #根据conce的sinkhorn矩阵，算出正例的prototype，村起来
        #根据正例的prototype，找到当前query中最能代表proto的那个patch，然后根据这个正例的conce矩阵，给定负例对应的权重，加权求和算出负例的prototype
        #对正负例进行约束
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.l2_norm = Normalize(2)
        self.batch_dim_for_bmm = 1
        self.ot = OT
        self.neg_type = opt.negtype
    def select_granteen(self, Q, K, num_of_proto):
        
        similarity = torch.bmm(Q, K.transpose(2, 1))## B, Sample_num,  dim X B,  dim, Sample_num  --> B, Sample_num, Sample_num,
        attn = similarity.softmax(dim=2)
        prob = -torch.log(attn)
        prob = torch.where(torch.isinf(prob), torch.full_like(prob, 0), prob) #  B, Sample_num, Sample_num,
        attn = similarity.softmax(dim=2)
        entropy = torch.sum(torch.mul(attn, prob), dim=2)  # 计算得出熵，以熵作为信息量的评判标准，越低的就越代表有效的表征  -->  B, Sample_num
        _, index = torch.sort(entropy)
        patch_id = index[:, :num_of_proto] # 取前N个作为计算原型用的候选patch

        return entropy, patch_id

    def cal_prototype(self, feat_q, feat_k, mode = None):
        num_patches = feat_q.shape[1]
        num_of_proto = num_patches//2

        #对于绝对正确的正阳本patch，有两种思路：q与q算，k与k算；；另一种，q与k算，相似对最高的q才有资格成为proto
        if mode == 'bg':
            # 思路2： q和k算，纯按照相似度来算，够高才有资格成为正样本
            entropy, patch_id = self.select_granteen(feat_q, feat_k, num_of_proto)
        else:
        # 思路1：自己跟自己算，但是分为有GT指引的patch相似度评价，和面向生成结果的相似度评价，两种思路
            entropy, patch_id = self.select_granteen(feat_q, feat_q, num_of_proto)
        
        entropy_k, patch_id_k = self.select_granteen(feat_k, feat_k, num_of_proto)

        

        q_proto_gran = feat_q[torch.arange(self.batch_dim_for_bmm)[:, None], patch_id, :]
        q_weight = entropy[torch.arange(self.batch_dim_for_bmm)[:, None], patch_id].softmax(dim=1).unsqueeze(2)
        q_proto = torch.sum(torch.mul(q_weight, q_proto_gran), dim=1)


        k_proto_gran = feat_k[torch.arange(self.batch_dim_for_bmm)[:, None], patch_id, :]
        k_weight = entropy_k[torch.arange(self.batch_dim_for_bmm)[:, None], patch_id].softmax(dim=1).unsqueeze(2)
        k_proto = torch.sum(torch.mul(k_weight, k_proto_gran), dim=1)

        return q_proto, k_proto, q_proto_gran, k_proto_gran
    
    def get_nce_loss(self, feat_q, feat_k):
        

        # feat_q = feat_q.view(self.batch_dim_for_bmm, -1, dim)  ## B, Sample_num,  dim 
        # feat_k = feat_k.view(self.batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))## B, Sample_num,  dim X B,  dim, Sample_num  --> B, Sample_num, Sample_num,

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]  #仅对角线为1的EYE矩阵
        l_neg_curbatch.masked_fill_(diagonal, -10.0)  #B, Sample_num, Sample_num

        num_patches = feat_q.shape[1]
        l_pos = torch.bmm(feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))  # B*Sample_num, 1, dim   x  B*Sample_num,dim,1  --> B*Sample_num,1 (B*Sample_num:整个MINIBATCH内采样总数量)
        l_pos = l_pos.view(num_patches, 1)

        l_neg = l_neg_curbatch.view(-1, npatches) #B * Sample_num, Sample_num

        out = torch.cat((l_pos, l_neg), dim=1) / 0.07
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,device=feat_q.device)).mean()

        return loss
    def forward(self, feat_qs, feat_ks, i):
        # q: fake images k: gt
        feat_q, feat_k = feat_qs[0], feat_ks[0]
        feat_q_neg, feat_k_neg = feat_qs[1], feat_ks[1]
        feat_q_bg, feat_k_bg = feat_qs[2], feat_ks[2]
        num_patches = feat_q.shape[0]
        num_patches_bg = feat_q_neg.shape[0]
        dim = feat_q.shape[1]
        # Therefore, we will include the negatives from the entire minibatch.
        # if self.opt.nce_includes_all_negatives_from_minibatch:
        #     batch_dim_for_bmm = 1
        # else:

        # if num_patches % 2 != 0:
        #     print(i, feat_q.shape)
        # if self.opt.divco:
        #     batch_dim_for_bmm = self.opt.batchSize // len(self.opt.gpu_ids) * (self.opt.num_negative + 2)
        # else:
        #只能取1,因为对instance进行干预后。每一个实例的mask都不一样，minibatch内的patch数量不一致

        # print('feat_q', feat_q.shape)
        feat_q = feat_q.view(self.batch_dim_for_bmm, -1, dim)  ## B, Sample_num,  dim 
        feat_k = feat_k.view(self.batch_dim_for_bmm, -1, dim)
        feat_q_neg = feat_q_neg.view(self.batch_dim_for_bmm, -1, dim)  ## B, Sample_num,  dim 
        feat_k_neg = feat_k_neg.view(self.batch_dim_for_bmm, -1, dim)
        feat_q_bg = feat_q_bg.view(self.batch_dim_for_bmm, -1, dim)  ## B, Sample_num,  dim 
        feat_k_bg = feat_k_bg.view(self.batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        
        q_pos_proto, k_pos_proto, q_pos_proto_gran, k_pos_proto_gran = self.cal_prototype(feat_q, feat_k)
        
        q_bg_proto, k_bg_proto, q_bg_proto_gran, k_bg_proto_gran = self.cal_prototype(feat_q_bg, feat_k_bg, mode = 'bg')

        if self.ot:
            if self.neg_type == 'frompos' or  feat_q_bg.shape[1]  == 0 :
            #思路1：根据正例的prototype，找到当前query中最能代表proto的那个patch，然后根据这个正例的conce矩阵，给定负例对应的权重，加权求和算出负例的prototype
                ot_q = feat_q.view(self.batch_dim_for_bmm, -1, dim)
                ot_k = feat_q_neg.view(self.batch_dim_for_bmm, -1, dim)
                similarity_sinkhorn = intervOT(ot_k, ot_q, eps=1.0, max_iter=50, cost_type = self.opt.cost_type)

                proto_simil = torch.bmm(q_pos_proto.view(self.batch_dim_for_bmm, -1, dim), feat_q.transpose(2, 1)) ## B, 1,  dim X B,  dim, Sample_num  --> B, 1, Sample_num,
                _, index_p = torch.max(proto_simil, 2)

                neg_weight = similarity_sinkhorn[torch.arange(self.batch_dim_for_bmm)[:, None], index_p, :].transpose(2, 1)
                q_neg_proto_gran = feat_q_neg
                k_neg_proto_gran =  feat_k_neg
                q_neg_proto = torch.sum(torch.mul(neg_weight, feat_q_neg), dim=1)
                k_neg_proto = torch.sum(torch.mul(neg_weight, feat_k_neg), dim=1)
            elif self.neg_type == 'frombg':
            #思路2：根据背景的prototype，找到当前query中最能代表proto的那个背景patch，然后根据这个背景与负例的conce矩阵，找到背景patch的那一列，给定负例对应的权重，加权求和算出负例的prototype
                ot_q = feat_q_bg.view(self.batch_dim_for_bmm, -1, dim)
                ot_k = feat_q_neg.view(self.batch_dim_for_bmm, -1, dim)
                similarity_sinkhorn = intervOT(ot_k, ot_q, eps=1.0, max_iter=50, cost_type = self.opt.cost_type)

                proto_simil = torch.bmm(q_bg_proto.view(self.batch_dim_for_bmm, -1, dim), feat_q_bg.transpose(2, 1)) ## B, 1,  dim X B,  dim, Sample_num  --> B, 1, bg_num,
                _, index_p = torch.max(proto_simil, 2)

                neg_weight = similarity_sinkhorn[torch.arange(self.batch_dim_for_bmm)[:, None], index_p, :].transpose(2, 1)
                q_neg_proto_gran = feat_q_neg
                k_neg_proto_gran =  feat_k_neg

                q_neg_proto = torch.sum(torch.mul(neg_weight, feat_q_neg), dim=1)
                k_neg_proto = torch.sum(torch.mul(neg_weight, feat_k_neg), dim=1)
            else:
            #思路3：根据prototype，直接计算相似度并加权 
                proto_simil = torch.bmm(q_bg_proto.view(self.batch_dim_for_bmm, -1, dim), feat_q_neg.transpose(2, 1)) 
                _, index = torch.sort(proto_simil)

                patch_id = index[:, :feat_q_neg.shape[1]] # 取前N个作为计算原型用的候选patch

                q_neg_proto_gran = feat_q_neg[torch.arange(self.batch_dim_for_bmm)[:, None], patch_id, :]
                q_weight = proto_simil[torch.arange(self.batch_dim_for_bmm)[:, None], patch_id].softmax(dim=1).unsqueeze(2)
                q_neg_proto = torch.sum(torch.mul(q_weight, q_neg_proto_gran), dim=1)

                k_neg_proto_gran = feat_k[torch.arange(self.batch_dim_for_bmm)[:, None], patch_id, :]
                # k_weight = entropy_k[torch.arange(self.batch_dim_for_bmm)[:, None], patch_id].softmax(dim=1).unsqueeze(2)
                k_neg_proto = torch.sum(torch.mul(q_weight, k_neg_proto_gran), dim=1)


        else:
            q_neg_proto, k_neg_proto, q_neg_proto_gran, k_neg_proto_gran = self.cal_prototype(feat_q_neg, feat_k_neg)



        # ot_q = feat_q.view(self.batch_dim_for_bmm, -1, dim)
        # ot_k = feat_k.view(self.batch_dim_for_bmm, -1, dim).detach()

        # f = intervOT(ot_q, ot_k, eps=1.0, max_iter=50, cost_type = self.opt.cost_type)
        loss = 0
        if q_pos_proto_gran.shape[1] >0 :
            loss += self.get_nce_loss( q_pos_proto_gran, k_pos_proto_gran)
        if q_neg_proto_gran.shape[1] >0 :
            loss += self.get_nce_loss( q_neg_proto_gran, k_neg_proto_gran)
        if q_bg_proto_gran.shape[1] >0 :
            loss += self.get_nce_loss( q_bg_proto_gran, k_bg_proto_gran)
        # loss = loss_pos + loss_neg  + loss_bg

        # feat_k = feat_k.detach()
        # feat_k_neg = feat_k_neg.detach()
        # # pos logit
        # l_pos = torch.bmm(feat_q_neg.view(num_patches_bg, 1, -1), feat_k_neg.view(num_patches_bg, -1, 1))
        # # print(i,'pos:',l_pos.shape, 'f_max',f_max.shape)
        # # if i == 3:
        # #     l_pos = l_pos.view(num_patches, 1) + torch.log(f_max) * 0.07
        # # else:
        # l_pos = l_pos.view(num_patches_bg, 1)

        # # reshape features to batch size
        

        # feat_q = feat_q.view(self.batch_dim_for_bmm, -1, dim)
        # feat_k = feat_k.view(self.batch_dim_for_bmm, -1, dim)
        # npatches = feat_q.size(1)
        # l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # # print(i, 'feat_q.feat_k', feat_q.shape, feat_k.shape)
        # # diagonal entries are similarity between same features, and hence meaningless.
        # # just fill the diagonal with very small number, which is exp(-10) and almost zero
        # diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        # l_neg_curbatch.masked_fill_(diagonal, -10.0)
        # l_neg = l_neg_curbatch.view(-1, npatches)/ 0.07
        # # print(i,'neg:',l_neg.shape, 'diagonal',diagonal.shape)

        # # out = torch.cat((l_pos, l_neg), dim=1) / 0.07
        # # print(i,'out:',out.shape)
        # loss = self.cross_entropy_loss(l_neg, torch.zeros(l_neg.size(0), dtype=torch.long, device=feat_q.device)).mean() \
        #     + self.bce_loss(l_pos, torch.ones(l_pos.size(), dtype=torch.float, device=feat_q.device)).mean()

        return loss
