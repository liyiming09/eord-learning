"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from sqlite3 import InterfaceError
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
import util.util as util


class LatentCodesDiscriminator(nn.Module):
    def __init__(self, style_dim, n_mlp):
        super().__init__()

        self.style_dim = style_dim

        layers = []
        layers.append(
                nn.Linear(1024, style_dim)
            )
        layers.append(nn.LeakyReLU(0.2))
        for i in range(n_mlp-2):
            layers.append(
                nn.Linear(style_dim, style_dim)
            )
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(style_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, w):
        return self.mlp(w)

def MLP(dim, projection_size, hidden_size=2048):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=512):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )


class SesameMultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='sesame_n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt, input_nc = None):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt, input_nc)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt, input_nc = None):
        subarch = opt.netD_subarch
        if subarch == 'sesame_n_layer':
            netD = SesameNLayerDiscriminator(opt, input_nc)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the SESAME discriminator with the specified arguments.
class SesameNLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt, input_nc=None):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        if input_nc is None:
            input_nc = self.compute_D_input_nc(opt)

        branch = []
        sizes = (input_nc - 3, 3) 
        original_nf = nf
        for input_nc in sizes: 
            nf = original_nf
            norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
            sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                         nn.LeakyReLU(0.2, False)]]

            for n in range(1, opt.n_layers_D):
                nf_prev = nf
                nf = min(nf * 2, 512)
                stride = 1 if n == opt.n_layers_D - 1 else 2
                sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                                   stride=stride, padding=padw)),
                              nn.LeakyReLU(0.2, False)
                              ]]

            branch.append(sequence)
            
        sem_sequence = nn.ModuleList()
        for n in range(len(branch[0])):
            sem_sequence.append(nn.Sequential(*branch[0][n]))
        self.sem_sequence = nn.Sequential(*sem_sequence)

        sequence = branch[1]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        self.img_sequence = nn.ModuleList()
        for n in range(len(sequence)):
            self.img_sequence.append(nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        label_nc = opt.label_nc
        input_nc = label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        if not opt.no_inpaint:
            input_nc += 1
            
        return input_nc

    def forward(self, input):
        img, sem = input[:,-3:], input[:,:-3]
        sem_results = self.sem_sequence(sem)
        results = [img]
        for submodel in self.img_sequence[:-1]:
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        intermediate_output = self.my_dot(intermediate_output, sem_results)
        results.append(self.img_sequence[-1](intermediate_output))

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]

    def my_dot(self, x, y):
        return x + x * y.sum(1).unsqueeze(1)


class DivcoMultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='divco_n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt, input_nc = None):
        super().__init__()
        self.opt = opt
        self.feat_dim = 128
        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt, input_nc)
            self.add_module('discriminator_%d' % i, subnetD)
        
        # self.conv_feat = nn.Conv2d(self.feat_dim*2, self.feat_dim, 1, 1, 0)

    def create_single_discriminator(self, opt, input_nc = None):
        subarch = opt.netD_subarch
        if subarch == 'divco_n_layer':
            netD = DivcoNLayerDiscriminator(opt, input_nc)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input, enc_feat = False):
        result = []
        feats = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        
        for name, D in self.named_children():
            if enc_feat:
                out, feat = D(input, enc_feat)
                feats.append(feat)
            else:
                out = D(input, enc_feat)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)
        if enc_feat:
            # out_feat = self.conv_feat(torch.cat(feats,1))
            return result, feats[0]
        else:
            return result


# Defines the SESAME discriminator with the specified arguments.
class DivcoNLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt, input_nc=None):
        super().__init__()
        self.opt = opt
        self.base = self.opt.batchSize//len(self.opt.gpu_ids) #每一个基础单元内有几个实例
        feat_dim = 128
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        if input_nc is None:
            input_nc = self.compute_D_input_nc(opt)

        branch = []
        sizes = (input_nc - 3, 3) 
        original_nf = nf
        for input_nc in sizes: 
            nf = original_nf
            norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
            sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                         nn.LeakyReLU(0.2, False)]]

            for n in range(1, opt.n_layers_D):
                nf_prev = nf
                nf = min(nf * 2, 512)
                stride = 1 if n == opt.n_layers_D - 1 else 2
                sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                                   stride=stride, padding=padw)),
                              nn.LeakyReLU(0.2, False)
                              ]]

            branch.append(sequence)
            
        sem_sequence = nn.ModuleList()
        for n in range(len(branch[0])):
            sem_sequence.append(nn.Sequential(*branch[0][n]))
        self.sem_sequence = nn.Sequential(*sem_sequence)

        sequence = branch[1]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        self.img_sequence = nn.ModuleList()
        for n in range(len(sequence)):
            self.img_sequence.append(nn.Sequential(*sequence[n]))

        self.conv_feat = nn.Conv2d(512+256+128+64, feat_dim, 1, 1, 0)
        self.feat_bn = nn.BatchNorm2d(feat_dim)
        self.predictor = MLP(feat_dim,feat_dim)
    def compute_D_input_nc(self, opt):
        label_nc = opt.label_nc
        input_nc = label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        if not opt.no_inpaint:
            input_nc += 1
            
        return input_nc

    def forward(self, input, enc_feat = False):
        # print('input:',input.shape)
        img, sem = input[:,-3:], input[:,:-3]
        sem_results = self.sem_sequence(sem)
        results = [img]
        if enc_feat:
            feat = []
        for submodel in self.img_sequence[:-1]:
            intermediate_output = submodel(results[-1])
            
            results.append(intermediate_output)
            if enc_feat:
                b = torch.nn.functional.adaptive_avg_pool2d(intermediate_output,(1,1))
                feat.append(b)
        if enc_feat:
            mix_feat = torch.cat(feat,1)
            out_feat = self.conv_feat(mix_feat)
            out_feat = self.feat_bn(out_feat)
            src = out_feat[0:1*self.base,...].clone().view(self.base,-1)# shape:1x128
            # print(src.shape)
            src = self.predictor(src)
            pos = out_feat[1*self.base:2*self.base,...].clone().view(self.base,-1)
            out = torch.cat((src,pos),0)
            # out_feat = torch.nn.functional.normalize(out_feat, dim=1)
        intermediate_output = self.my_dot(intermediate_output, sem_results)
        results.append(self.img_sequence[-1](intermediate_output))

        get_intermediate_features = not self.opt.no_ganFeat_loss

        if enc_feat:
            
            if get_intermediate_features:
                # print('result:',results[1].shape)
                return results[1:], out#out_feat[:input.shape[0]//2]
            else:
                return results[-1], out#out_feat[:input.shape[0]//2]
        else:
            if get_intermediate_features:
                return results[1:]
            else:
                return results[-1]

    def my_dot(self, x, y):
        return x + x * y.sum(1).unsqueeze(1)



class LearnEorDDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt, label_nc=None):
        super().__init__()
        self.opt = opt
        self.base = self.opt.batchSize//len(self.opt.gpu_ids) #每一个基础单元内有几个实例
        feat_dim = 128
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        if label_nc is None:
            label_nc = 1
            # input_nc = self.compute_D_input_nc(opt)

        branch = []
        sizes = (label_nc, 3) 
        original_nf = nf
        for input_nc in sizes: 
            nf = original_nf
            norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
            sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                         nn.LeakyReLU(0.2, False)]]

            for n in range(1, opt.n_layers_D):
                nf_prev = nf
                nf = min(nf * 2, 512)
                stride = 1 if n == opt.n_layers_D - 1 else 2
                sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                                   stride=stride, padding=padw)),
                              nn.LeakyReLU(0.2, False)
                              ]]

            branch.append(sequence)
            
        sem_sequence = nn.ModuleList()
        for n in range(len(branch[0])):
            sem_sequence.append(nn.Sequential(*branch[0][n]))
        self.sem_sequence = nn.Sequential(*sem_sequence)

        sequence = branch[1]
        sequence += [[norm_layer(nn.Conv2d(nf, feat_dim, kernel_size=kw,
                                                   stride=stride, padding=padw)),
                              nn.LeakyReLU(0.2, False)
                              ]]
                              


        # We divide the layers into groups to extract intermediate layer outputs
        self.img_sequence = nn.ModuleList()
        for n in range(len(sequence)):
            self.img_sequence.append(nn.Sequential(*sequence[n]))
        self.interv_conv = nn.Sequential(norm_layer(nn.Conv2d(feat_dim, feat_dim, kernel_size=kw,
                                                   stride=stride, padding=padw)),
                              nn.LeakyReLU(0.2, False)

        )
        self.interv_pool = nn.AdaptiveAvgPool2d(1)
        self.interv_head = nn.Conv2d(feat_dim,1, kernel_size=kw, stride=1, padding=padw)
        self.cls_conv =  nn.Sequential(norm_layer(nn.Conv2d(feat_dim, feat_dim, kernel_size=kw,
                                                   stride=stride, padding=padw)),
                              nn.LeakyReLU(0.2, False),
                              nn.AdaptiveAvgPool2d(1)
        )
        self.cls_head = nn.Linear(feat_dim, opt.label_nc)

        # self.conv_feat = nn.Conv2d(512+256+128+64, feat_dim, 1, 1, 0)
        # self.feat_bn = nn.BatchNorm2d(feat_dim)
        # self.predictor = MLP(feat_dim,feat_dim)
    def compute_D_input_nc(self, opt):
        label_nc = opt.label_nc
        input_nc = label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        if not opt.no_inpaint:
            input_nc += 1
            
        return input_nc

    def forward(self, input, enc_feat = False, is_generate = False):
        # print('input:',input.shape)
        img, sem = input[:,-3:], input[:,:-3]
        sem_results = self.sem_sequence(sem)
        results = [img]

        for submodel in self.img_sequence[:-1]:
            intermediate_output = submodel(results[-1])
            
            results.append(intermediate_output)
            # if enc_feat:
            #     b = torch.nn.functional.adaptive_avg_pool2d(intermediate_output,(1,1))
            #     feat.append(b)
        # if enc_feat:
        #     mix_feat = torch.cat(feat,1)
        #     out_feat = self.conv_feat(mix_feat)
        #     out_feat = self.feat_bn(out_feat)
        #     src = out_feat[0:1*self.base,...].clone().view(self.base,-1)# shape:1x128
        #     # print(src.shape)
        #     src = self.predictor(src)
        #     pos = out_feat[1*self.base:2*self.base,...].clone().view(self.base,-1)
        #     out = torch.cat((src,pos),0)
            # out_feat = torch.nn.functional.normalize(out_feat, dim=1)
        intermediate_output = self.my_dot(intermediate_output, sem_results)
        results.append(self.img_sequence[-1](intermediate_output))
        mix_feature = results[-1]
        interv_feature = self.interv_conv(mix_feature)
        interv_vector = self.interv_pool(interv_feature).view(interv_feature.size(0),-1)
        interv_res = self.interv_head(interv_feature)

        cls_vector = self.cls_conv(mix_feature).view(mix_feature.size(0),-1)
        cls_res = self.cls_head(cls_vector)

        if is_generate:
            return [interv_res]
        else:
            return [cls_res, interv_res, cls_vector, interv_vector]

    def my_dot(self, x, y):
        return x + x * y.sum(1).unsqueeze(1)


class OnlyEDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt, label_nc=None):
        super().__init__()
        self.opt = opt
        self.base = self.opt.batchSize//len(self.opt.gpu_ids) #每一个基础单元内有几个实例
        feat_dim = 128
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        if label_nc is None:
            label_nc = 1

        original_nf = nf
        input_nc = label_nc
        nf = original_nf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                        nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                                stride=stride, padding=padw)),
                            nn.LeakyReLU(0.2, False)
                            ]]


        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        sem_sequence = nn.ModuleList()
        for n in range(len(sequence)):
            sem_sequence.append(nn.Sequential(*sequence[n]))                 
        self.sem_sequence = nn.Sequential(*sem_sequence)

        # self.conv_feat = nn.Conv2d(512+256+128+64, feat_dim, 1, 1, 0)
        # self.feat_bn = nn.BatchNorm2d(feat_dim)
        # self.predictor = MLP(feat_dim,feat_dim)

    def forward(self, input, enc_feat = False, is_generate = False):
        # print('input:',input.shape)
        sem_results = self.sem_sequence(input)


        return sem_results


class SesameMultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='sesame_n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt, input_nc = None):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt, input_nc)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt, input_nc = None):
        subarch = opt.netD_subarch
        if subarch == 'sesame_n_layer':
            netD = SesameNLayerDiscriminator(opt, input_nc)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# # Defines the patchGAN discriminator with the specified arguments.
# class pix2pixDiscriminator(BaseNetwork):
#     @staticmethod
#     def modify_commandline_options(parser, is_train):
#         parser.add_argument('--n_layers_D', type=int, default=4,
#                             help='# layers in each discriminator')
#         return parser

#     def __init__(self, opt, input_nc=None):
#         super().__init__()
#         self.opt = opt

#         kw = 4
#         padw = int(np.ceil((kw - 1.0) / 2))
#         nf = opt.ndf
#         if input_nc is None:
#             input_nc = self.compute_D_input_nc(opt)

#         branch = []
#         sizes = (input_nc - 3, 3) 
#         original_nf = nf
#         for input_nc in sizes: 
#             nf = original_nf
#             norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
#             sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
#                          nn.LeakyReLU(0.2, False)]]

#             for n in range(1, opt.n_layers_D):
#                 nf_prev = nf
#                 nf = min(nf * 2, 512)
#                 stride = 1 if n == opt.n_layers_D - 1 else 2
#                 sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
#                                                    stride=stride, padding=padw)),
#                               nn.LeakyReLU(0.2, False)
#                               ]]

#             branch.append(sequence)
            
#         sem_sequence = nn.ModuleList()
#         for n in range(len(branch[0])):
#             sem_sequence.append(nn.Sequential(*branch[0][n]))
#         self.sem_sequence = nn.Sequential(*sem_sequence)

#         sequence = branch[1]
#         sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

#         # We divide the layers into groups to extract intermediate layer outputs
#         self.img_sequence = nn.ModuleList()
#         for n in range(len(sequence)):
#             self.img_sequence.append(nn.Sequential(*sequence[n]))

#     def compute_D_input_nc(self, opt):
#         label_nc = opt.label_nc
#         input_nc = label_nc + opt.output_nc
#         if opt.contain_dontcare_label:
#             input_nc += 1
#         if not opt.no_instance:
#             input_nc += 1
#         if not opt.no_inpaint:
#             input_nc += 1
            
#         return input_nc

#     def forward(self, input):
#         img, sem = input[:,-3:], input[:,:-3]
#         sem_results = self.sem_sequence(sem)
#         results = [img]
#         for submodel in self.img_sequence[:-1]:
#             intermediate_output = submodel(results[-1])
#             results.append(intermediate_output)

#         intermediate_output = self.my_dot(intermediate_output, sem_results)
#         results.append(self.img_sequence[-1](intermediate_output))

#         get_intermediate_features = not self.opt.no_ganFeat_loss
#         if get_intermediate_features:
#             return results[1:]
#         else:
#             return results[-1]

#     def my_dot(self, x, y):
#         return x + x * y.sum(1).unsqueeze(1)

import functools
class pix2pixDiscriminator(BaseNetwork):
    """Defines a PatchGAN discriminator"""

    def __init__(self, opt, input_nc = None, n_layers=3, norm_layer=nn.InstanceNorm2d,use_sigmoid=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(pix2pixDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1

        ndf = opt.ndf
        if input_nc is None:
            input_nc = self.compute_D_input_nc(opt)

        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def compute_D_input_nc(self, opt):
        label_nc = opt.label_nc
        input_nc = label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        if not opt.no_inpaint:
            input_nc += 1
            
        return input_nc

    def forward(self, input):
        """Standard forward."""
        return self.model(input)