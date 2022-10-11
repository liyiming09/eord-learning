"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.networks.conce import CoNCELoss, PatchNCELoss
# from models.networks.architecture import VGG19
import torch.distributed as dist

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                # assert target_is_real, "The generator's hinge loss must be aiming for real"
                if target_is_real:
                    loss = -torch.mean(input)
                else:
                    target_tensor = self.get_target_tensor(input, target_is_real)
                    return F.mse_loss(input, target_tensor)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class oldVGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class DivcoLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, opt):
        super(DivcoLoss, self).__init__()

        self.criterion = torch.nn.L1Loss()
        self.opt = opt
        gpu_ids = self.opt.gpu_ids
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()


    def compute_contrastive_loss(self, feat_q, feat_k):
        out = torch.mm(feat_q, feat_k.transpose(1,0)) / self.opt.tau
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        # print(out.dtype)
        return loss

    def __call__(self, feat):
        loss_contra = 0.0
        # seprate features
        # for cur in range(self.opt.num_D):
        cur_feat = feat
    # Compute loss
        for i in range(self.opt.batchSize//len(self.opt.gpu_ids)):
            # print(cur_feat[i:cur_feat.shape[0]:(self.opt.batchSize//len(self.opt.gpu_ids))].shape)
            # print(cur_feat.shape[0],cur_feat.shape[0], (self.opt.batchSize//len(self.opt.gpu_ids)))
            logits = cur_feat[i:cur_feat.shape[0]:(self.opt.batchSize//len(self.opt.gpu_ids))].reshape(self.opt.num_negative+2, -1)
            # print(logits.shape)
            if self.opt.featnorm:
                logits = logits / torch.norm(logits, p=2, dim=1, keepdim=True)
            loss_contra += self.compute_contrastive_loss(logits[1:2], logits[2:])


        return loss_contra/(self.opt.batchSize//len(self.opt.gpu_ids))

class ByolLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, opt):
        super(ByolLoss, self).__init__()

        self.criterion = torch.nn.L1Loss()
        self.opt = opt
        self.base = self.opt.batchSize//len(self.opt.gpu_ids) #每一个基础单元内有几个实例
        gpu_ids = self.opt.gpu_ids
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()


    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)
    def loss_cos(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        cos = 1 - F.cosine_similarity(x,y,dim=-1)
        return cos
    def compute_contrastive_loss(self, feat_q, feat_k):
        out = torch.mm(feat_q, feat_k.transpose(1,0)) / self.opt.tau
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        # print(out.dtype)
        return loss

    def __call__(self, feat):
        src = feat[0:1*self.base,...]
        pos = feat[1*self.base:2*self.base,...]
        # print(src.shape,pos.shape)
        loss_contra = self.loss_cos(src,pos)

        return loss_contra

class CosLoss(nn.Module):
    def __init__(self):
        super(CosLoss, self).__init__()

        self.criterion = torch.nn.L1Loss()
        # self.opt = opt
        # self.base = self.opt.batchSize//len(self.opt.gpu_ids) #每一个基础单元内有几个实例
    
    def loss_cos(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        cos = F.cosine_similarity(x,y,dim=-1)
        return cos

    def __call__(self, x,y):
        cos_loss = self.loss_cos(x,y)

        return cos_loss


class ModeSeekingLoss(nn.Module):


    def __init__(self, opt):
        super(ModeSeekingLoss, self).__init__()

        self.criterion = torch.nn.SmoothL1Loss()
        self.opt = opt
        gpu_ids = self.opt.gpu_ids
        self.simi_loss = torch.nn.MSELoss()
        self.eps = 1 * 1e-5

    def loss_cos(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        cos = 1 - F.cosine_similarity(x,y,dim=-1)
        # print(x.shape,y.shape,cos)
        return cos


    def compute_modeseeking_loss(self, feat, semantic):
        # for k in range(feat.shape[0]):

        # print('feat', feat.shape,feat[1:].shape,feat[0:1].shape)
        # print('semantic', semantic.shape,semantic[1:].shape,semantic[0:1].shape)
        # out = self.criterion(feat[1:],feat[0:1])/self.criterion(semantic[1:],semantic[0:1])
        feat_vector = torch.mean(torch.abs(feat[1:]-feat[0:1]),dim = 1)
        sem_vector = torch.mean(torch.abs(semantic[1:]-semantic[0:1]),dim = 1)
        # print(feat_vector, sem_vector)
        # similarity = feat_vector/(sem_vector + self.eps)
        # loss = self.simi_loss(similarity, torch.ones(similarity.size(0), dtype=torch.float,
        #                                             device=feat.device))
        loss = self.loss_cos(feat_vector, sem_vector)


        # print(out.dtype)
        return loss

    def __call__(self, feat, semantic_map):
        loss_ms = 0.0
        # seprate features
        # for cur in range(self.opt.num_D):
        cur_feat = feat
        # print(cur_feat.shape,semantic_map.shape)
    # Compute loss
        for i in range(self.opt.batchSize//len(self.opt.gpu_ids)):
            # print(cur_feat[i:cur_feat.shape[0]:(self.opt.batchSize//len(self.opt.gpu_ids))].shape)
            # print(cur_feat.shape[0],cur_feat.shape[0], (self.opt.batchSize//len(self.opt.gpu_ids)))
            # 前半部分为feak，后一半为real，在D中已经将real部分舍去了。将fake部分的按照single source的顺序挑选出来
            feat_ = cur_feat[i:cur_feat.shape[0]:(self.opt.batchSize//len(self.opt.gpu_ids))].reshape(self.opt.num_negative+2, -1)
            semantic = semantic_map[i:cur_feat.shape[0]:(self.opt.batchSize//len(self.opt.gpu_ids)),...].reshape(self.opt.num_negative+2, -1)
            # print(1111,feat_.shape, semantic.shape )
            # print(logits.shape)
            if self.opt.featnorm:
                logits = logits / torch.norm(logits, p=2, dim=1, keepdim=True)
            loss_ms += self.compute_modeseeking_loss(feat_, semantic)


        return loss_ms/(self.opt.batchSize//len(self.opt.gpu_ids))

class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, gpu_ids):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss



class VGGLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, gpu_ids, weights=[1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]):
        super(VGGLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss



class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = torchvision.models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out

class EffectLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(EffectLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, semantics, target_is_real):
        # print(input.shape,semantics.shape)
        new_map = torch.nn.functional.interpolate(semantics, size=(input.shape[2],input.shape[3]), mode='bilinear',align_corners=True)
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
                
            return self.real_label_tensor.expand_as(input) * new_map
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input) * new_map

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, semantics, target_is_real, for_discriminator=True):
        # if self.gan_mode == 'original':  # cross entropy loss
        #     target_tensor = self.get_target_tensor(input, target_is_real)
        #     loss = F.binary_cross_entropy_with_logits(input, target_tensor)
        #     return loss
        # elif self.gan_mode == 'ls':
        #     target_tensor = self.get_target_tensor(input, target_is_real)
        #     return F.mse_loss(input, target_tensor)
        # elif self.gan_mode == 'hinge':
        #     if for_discriminator:
        #         if target_is_real:
        #             minval = torch.min(input - 1, self.get_zero_tensor(input))
        #             loss = -torch.mean(minval)
        #         else:
        #             minval = torch.min(-input - 1, self.get_zero_tensor(input))
        #             loss = -torch.mean(minval)
        #     else:
        #         assert target_is_real, "The generator's hinge loss must be aiming for real"
        #         loss = -torch.mean(input)
        #     return loss
        # else:
        #     # wgan
        #     if target_is_real:
        #         return -input.mean()
        #     else:
        #         return input.mean()
        target_tensor = self.get_target_tensor(input, semantics, target_is_real)
        return F.mse_loss(input, target_tensor, reduction='sum')/10000

    def __call__(self, input_neg,input_fake,  semantics, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input_neg, list):
            loss = 0
            for pred_neg,pred_fake in zip(input_neg,input_fake):
                if isinstance(pred_neg, list):
                    pred_neg = pred_neg[0]
                    pred_fake = pred_fake[0]
                    # if dist.get_rank() == 0: print(pred_neg==pred_fake)
                    pred_i = -pred_neg + pred_fake
                loss_tensor = self.loss(pred_i, semantics, target_is_real, for_discriminator)
                # if dist.get_rank() == 0: print(loss_tensor.shape, loss_tensor)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input_neg)
        else:
            pred = -input_neg + input_fake
            return self.loss(pred, semantics, target_is_real, for_discriminator)



# Perceptual loss that uses a pretrained VGG network
class MoNCELoss(nn.Module):
    def __init__(self, opt, netF):
        super(MoNCELoss, self).__init__()
        # self.opt = opt
        self.vgg = VGG19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = CoNCELoss(opt)
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.netF = netF
        self.patch_num = 128


    def forward(self, x, y, bbox):
        # print('img.shape:',x.shape,y.shape)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        
        x_vggs = [x_vgg['relu1_1'],x_vgg['relu2_1'],x_vgg['relu3_1'],x_vgg['relu4_1']]
        y_vggs = [y_vgg['relu1_1'],y_vgg['relu2_1'],y_vgg['relu3_1'],y_vgg['relu4_1']]
        feat_k_pool, sample_ids = self.netF(y_vggs, bbox, self.patch_num, None)
        feat_q_pool, _ = self.netF(x_vggs, bbox, self.patch_num, sample_ids) # synthesis
            

        loss = 0
        for i in range(len(x_vggs)):
            # print(i, 'input:',feat_q_pool[i].shape, feat_k_pool[i].shape)
            if feat_q_pool[i].shape[0] == 0:
                print('too small region')
                continue
            loss += self.weights[i] * self.criterion(feat_q_pool[i], feat_k_pool[i], i)

        # print('******', loss.mean())
        # 1/0


        return loss

class PatchLoss(nn.Module):
    def __init__(self, opt, netF):
        super(PatchLoss, self).__init__()
        # self.opt = opt
        self.vgg = VGG19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = PatchNCELoss(opt)
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.netF = netF
        self.patch_num = 128


    def forward(self, x, y, bbox):
        # print('img.shape:',x.shape,y.shape)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        
        x_vggs = [x_vgg['relu1_1'],x_vgg['relu2_1'],x_vgg['relu3_1'],x_vgg['relu4_1']]
        y_vggs = [y_vgg['relu1_1'],y_vgg['relu2_1'],y_vgg['relu3_1'],y_vgg['relu4_1']]
        feat_k_pool, sample_ids = self.netF(y_vggs, bbox, self.patch_num, None)
        feat_q_pool, _ = self.netF(x_vggs, bbox, self.patch_num, sample_ids) # synthesis
            

        loss = 0
        for i in range(len(x_vggs)):
            # print(i, 'input:',feat_q_pool[i].shape, feat_k_pool[i].shape)
            if feat_q_pool[i].shape[0] == 0:
                print('too small region')
                continue
            loss += self.weights[i] * self.criterion(feat_q_pool[i], feat_k_pool[i])

        # print('******', loss.mean())
        # 1/0


        return loss