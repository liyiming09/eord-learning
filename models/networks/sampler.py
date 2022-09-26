""" Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class PatchSampleFSampler(BaseNetwork):
    def __init__(self,opt):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleFSampler, self).__init__()
        self.l2norm = Normalize(2)
        self.opt  = opt
        self.nc = 256#self.opt.nc  # hard-coded
        self.finesize = 256
    # def create_mlp(self, feats):
    #     for mlp_id, feat in enumerate(feats):
    #         input_nc = feat.shape[1]
    #         mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
    #         if len(self.gpu_ids) > 0:
    #             mlp.cuda()
    #         setattr(self, 'mlp_%d' % mlp_id, mlp)
    #     # init_net(self, self.init_type, self.init_gain, self.gpu_ids)
    #     self.mlp_init = True

    def forward(self, feats, bbox, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        # if self.use_mlp and not self.mlp_init:
        #     self.create_mlp(feats)
        for feat_id, sfeat in enumerate(feats):
            B, H, W = sfeat.shape[0], sfeat.shape[2], sfeat.shape[3]
            if self.opt.eord:
                patch_wmin, patch_hmin, patch_wmax, patch_hmax = int(bbox[0]*W/self.finesize), int(bbox[1]*H/self.finesize), int(bbox[2]*W/self.finesize), int(bbox[3]*H/self.finesize)
                feat = sfeat[:,:,patch_hmin:patch_hmax, patch_wmin:patch_wmax].clone()
            else:
                feat = sfeat
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # B,H,W,C, B,L,C
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device).num_patch.
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1]).
            else:
                x_sample = feat_reshape
                patch_id = []
            # if self.use_mlp:
            #     mlp = getattr(self, 'mlp_%d' % feat_id)
            #     x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)
            # print(patch_ids==None, 'x_sample', feat_id,x_sample.shape)
            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids  


class BoxSampleFSampler(BaseNetwork):
    def __init__(self,opt):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(BoxSampleFSampler, self).__init__()
        self.l2norm = Normalize(2)
        self.opt  = opt
        self.nc = 256#self.opt.nc  # hard-coded
        self.finesize = 256
    # def create_mlp(self, feats):
    #     for mlp_id, feat in enumerate(feats):
    #         input_nc = feat.shape[1]
    #         mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
    #         if len(self.gpu_ids) > 0:
    #             mlp.cuda()
    #         setattr(self, 'mlp_%d' % mlp_id, mlp)
    #     # init_net(self, self.init_type, self.init_gain, self.gpu_ids)
    #     self.mlp_init = True

    def forward(self, feats, bbox, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        # if self.use_mlp and not self.mlp_init:
        #     self.create_mlp(feats)
        for feat_id, sfeat in enumerate(feats):
            B, H, W = sfeat.shape[0], sfeat.shape[2], sfeat.shape[3]
            x_sample_bs = []
            patch_id_bs = []
            for i in range(B):
                patch_wmin, patch_hmin, patch_wmax, patch_hmax = int(bbox[0][i]*W/self.finesize), int(bbox[1][i]*H/self.finesize), int(bbox[2][i]*W/self.finesize), int(bbox[3][i]*H/self.finesize)
                feat = sfeat[i:i+1,:,patch_hmin:patch_hmax, patch_wmin:patch_wmax].clone()
                feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # 1,H,W,C, 1,L,C
                
                if num_patches > 0:
                    if patch_ids is not None:
                        patch_id = patch_ids[feat_id][i]
                    else:
                        patch_id = torch.randperm(feat_reshape.shape[1])
                        # print('patch_id', patch_id.shape[0])
                        num_patches_flr = patch_id.shape[0]//B*B
                        patch_id = patch_id[:int(min(num_patches, num_patches_flr))]  # .to(patch_ids.device).num_patch.
                        patch_id_bs.append(patch_id)
                    x_sample_bs.append(feat_reshape[:, patch_id, :].flatten(0, 1))  # reshape(-1, x.shape[1]).
                    # print('single', i,x_sample_bs[-1].shape)
                else:
                    x_sample_bs.append(feat_reshape)
                    patch_id = []
            # if self.use_mlp:
            #     mlp = getattr(self, 'mlp_%d' % feat_id)
            #     x_sample = mlp(x_sample)
            return_ids.append(patch_id_bs)
            x_sample = torch.cat(x_sample_bs,0)  # cat in dimension-B
            x_sample = self.l2norm(x_sample)
            # print(patch_ids==None, 'x_sample', feat_id,x_sample.shape)
            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids