""".unsqueeze(0)
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from curses import A_ALTCHARSET
import cv2
import torchvision.transforms as t 
import models.networks as networks
import util.util as util
from random import randint, random
from modules.vgg import vgg16, vgg16_bn
from modules.resnet import resnet50, resnet101
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import numpy as np
from models.networks.sync_batchnorm import DataParallelWithCallback
import matplotlib.cm
from matplotlib.cm import ScalarMappable

class ClsModel(torch.nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        # self.netG, self.netD = self.initialize_networks(opt)
        self.netC = self.initialize_networksC(opt)

       
        

        # set loss functions
        if opt.isTrain:
            self.criterioncls = torch.nn.CrossEntropyLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, data, mode):
        input_semantics, real_image, masked_image = self.preprocess_input(data)


            
        self.invent_semantics = []
        self.base_size = self.opt.batchSize//len(self.opt.gpu_ids)
        for it in range(len(input_semantics)):
            tmp_semantics = []
            for k in range(input_semantics[it].shape[0]):
                tmp_semantics.append(input_semantics[it][k:k+1,[self.cls[k%self.base_size,:]]])
            self.invent_semantics.append(torch.cat(tmp_semantics,dim = 0))

        if mode == 'classifier':
            # self.set_requires_grad(self.netD, False)
            # print(1)
            loss, acc, total = self.compute_cls_loss(input_semantics, real_image)
            return loss, masked_image, input_semantics + self.invent_semantics, [acc, total]
        elif mode == 'backword':
            maps = self.generate_map()
            return  maps, masked_image, input_semantics + self.invent_semantics
        elif mode == 'gen_C_set':
            maps = self.generate_map()
            return  maps, masked_image, input_semantics + self.invent_semantics
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        C_params = list(self.netC.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            C_lr  = opt.lr
        else:
            beta1, beta2 = 0, 0.9
            C_lr  = opt.lr

        optimizer_C = torch.optim.Adam(C_params, lr=C_lr, betas=(beta1, beta2))
        return  optimizer_C

    def save(self, epoch):
        util.save_network(self.netC, 'C', epoch, self.opt)


    ############################################################################
    # Private helper methods
    ############################################################################
    
    def initialize_networksC(self, opt):
        netC = networks.define_C(opt)
        
        # netCD = networks.define_onlyED(opt) if opt.isTrain else None

        # local_rank = self.opt.local_rank
        # torch.cuda.set_device(local_rank)
        # dist.init_process_group(backend='nccl')
        # device = torch.device("cuda", local_rank)
        
        # if len(opt.gpu_ids) > 0:
        #     netC = DataParallelWithCallback(netC, device_ids=opt.gpu_ids)
        #     # netC = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netC).to(device)
        #     # netC = DDP(netC,find_unused_parameters=False,  device_ids=[local_rank], output_device=local_rank).cuda()
        #     netC = netC.cuda()
        #     # if opt.isTrain:
        #     #     netCD = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netCD).to(device)
        #     #     netCD = DDP(netCD,find_unused_parameters=False,  device_ids=[local_rank], output_device=local_rank).cuda()

        if not opt.isTrain or opt.continue_train:
            netC = util.load_network(netC, 'C', opt.which_epoch, opt)
            # if opt.isTrain:
            #     netCD = util.load_network(netCD, 'ED', opt.which_epoch, opt)
        return netC

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        self.cls = data['cls'].cuda()
        if self.opt.monce or self.opt.patchnce:
            self.bbox = data['bbox']
        if not self.opt.bbox:
            if self.use_gpu():
                label = data['label'].cuda()
                inst = data['instance'].cuda()
                image =  data['image'].cuda()
                masked = data['image'].cuda()
            else:
                label = data['label']
                inst = data['instance']
                image =  data['image']
                masked = data['image']
       
        else:
            label = data['label'].cuda()
            inst = data['inst'].cuda()
            image = data['image'].cuda()
            mask_in = data['mask_in'].cuda()
            mask_out = data['mask_out'].cuda()
            masked = data['image'].cuda()
  
        # Get Semantics
        input_semantics = self.get_semantics(label)

        if not self.opt.no_instance:
            inst_map = inst
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        if not self.opt.no_inpaint:
            if "mask" in data.keys():
                mask = data["mask"]
                if self.use_gpu():
                    mask = mask.cuda()

            elif self.opt.bbox:
                mask = 1 - mask_in#背景区域
                   
            else:
                # Cellural Mask used for AIM2020 Challenge on Image Extreme Inpainting
                mask = self.get_mask(image.size())

            masked =  image * (1-mask)
            assert input_semantics.sum(1).max() == 1
            assert input_semantics.sum(1).min() == 1

            if self.opt.segmentation_mask:
                
                input_semantics *= (1-mask)


            input_semantics = torch.cat([input_semantics,1-mask[:,0:1]],dim=1)#input_semantics+遮挡区域

            semantics = [input_semantics]
        # print(image.shape, image.device,'img')
        return semantics, image, masked

    def preprocess_input3(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        self.cls = data['cls'].cuda()

        self.bbox = data['bbox']
        if not self.opt.bbox:
            if self.use_gpu():
                label = data['label'].cuda()
                inst = data['instance'].cuda()
                image =  data['image'].cuda()
                masked = data['image'].cuda()
            else:
                label = data['label']
                inst = data['instance']
                image =  data['image']
                masked = data['image']
       
        else:
            label = data['label'].cuda()
            inst = data['inst'].cuda()
            image = data['image'].cuda()
            mask_in = data['mask_in'].cuda()
            mask_out = data['mask_out'].cuda()
            masked = data['image'].cuda()
        self.eord_flag = False

        if self.opt.eord  and (not self.eord_flag):
            pos_label_list = data['ped_label']
            pos_labels = torch.cat(pos_label_list, 0).long().cuda()
            neg_label_list = data['ned_label']
            neg_labels = torch.cat(neg_label_list, 0).long().cuda()

                
        # Get Semantics
        input_semantics = self.get_semantics(label)
        if self.opt.eord  and (not self.eord_flag):
            pos_input_semantics = self.get_semantics(pos_labels)
            neg_input_semantics = self.get_semantics(neg_labels)
            # input_semantics = torch.cat((input_semantics, extra_input_semantics), dim=0)
        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = inst
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        if not self.opt.no_inpaint:
            if "mask" in data.keys():
                mask = data["mask"]
                if self.use_gpu():
                    mask = mask.cuda()

            elif self.opt.bbox:
                mask = 1 - mask_in#背景区域
                   
            else:
                # Cellural Mask used for AIM2020 Challenge on Image Extreme Inpainting
                mask = self.get_mask(image.size())


            assert input_semantics.sum(1).max() == 1
            assert input_semantics.sum(1).min() == 1

            masked =  image * mask

            #if erode or dilate, repeat the other tensor so as to match the dimension of input_semantics

            if self.opt.segmentation_mask:
                
                input_semantics *= (1-mask)


            input_semantics = torch.cat([input_semantics,1-mask[:,0:1]],dim=1)#input_semantics+遮挡区域
            if self.opt.eord  and (not self.eord_flag):
                if self.opt.segmentation_mask:
                
                    pos_input_semantics *= (1-mask)
                    neg_input_semantics *= (1-mask)
                pos_input_semantics = torch.cat([pos_input_semantics,1-mask[:,0:1]],dim=1)#
                neg_input_semantics = torch.cat([neg_input_semantics,1-mask[:,0:1]],dim=1)#
                semantics = [input_semantics, pos_input_semantics, neg_input_semantics]
            else:
                semantics = input_semantics
        # print(image.shape, image.device,'img')
        return semantics, image, masked


    def compute_cls_loss(self, input_semantics, images):
        C_losses = {}
        self.output = self.generate_classification()
        C_losses['C_ce'] = self.criterioncls(self.output, self.cls.squeeze())

        _, predicted = torch.max(self.output.data, 1)
        correct = (predicted == self.cls.squeeze()).sum().item()
        total = self.output.shape[0]
        return C_losses, correct, total
    
    def generate_classification(self, ):
        pred = self.netC(self.invent_semantics[0])

        return pred
    
    def generate_map(self, ):
        pred = self.netC(self.invent_semantics[0])
        T, cls = self.compute_pred(pred)

        if self.opt.method == 'LRP':
            Res = self.netC.relprop(R = pred * T, alpha= 1).sum(dim=1, keepdim=True)
        else:
            RAP = self.netC.RAP_relprop(R=T)
            Res = (RAP).sum(dim=1, keepdim=True)
        heatmap = Res.permute(0, 2, 3, 1).data.cpu().numpy()
        score_map, score_map_rgb = self.visualize(heatmap.reshape([self.base_size, 256, 256, 1]))

        return [score_map, score_map_rgb]
    def visualize(self, relevances):

        score_map = relevances[0,:,:,0]

        #取原值+归一化并保存map图像
        maxn, minn = score_map.max(), score_map.min()
        n = len(relevances)
        heatmap = np.sum(relevances.reshape([n, 256, 256, 1]), axis=3)

        # final_map = (score_map-minn)/(maxn-minn)
        # final_map = final_map*255

        # #取绝对值+归一化并保存map图像

        # final_map = (np.abs(score_map))/(maxn)
        # final_map = final_map*255
        
        #取与均值的距离作为score，并保存map图像
        R = heatmap[0]
        R = R / np.max(np.abs(R)) # normalize to [-1,1] wrt to max relevance magnitude
        R = (R + 1.)/2.
        final_map = R*255
        # 彩色化处理
        
        
        final_map_rgb = self.hm_to_rgb(heatmap[0], scaling=3, cmap = 'seismic')
        return final_map, final_map_rgb
    def hm_to_rgb(self, R, scaling = 3, cmap = 'bwr', normalize = True):
        cmap = eval('matplotlib.cm.{}'.format(cmap))
        if normalize:
            R = R / np.max(np.abs(R)) # normalize to [-1,1] wrt to max relevance magnitude
            R = (R + 1.)/2. # shift/normalize to [0,1] for color mapping
        R = R
        # R = enlarge_image(R, scaling)
        rgb = cmap(R.flatten())[...,0:3].reshape([R.shape[0],R.shape[1],3])
        return rgb

    def compute_pred(self, output):
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        # print('Pred cls : '+str(pred))
        T = pred.squeeze().cpu().numpy()
        T = np.expand_dims(T, 0)
        T = (T[:, np.newaxis] == np.arange(35)) * 1.0
        T = torch.from_numpy(T).type(torch.FloatTensor)
        Tt = T.cuda()
        return Tt, pred

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1]).byte()
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1]).byte()
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).byte()
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).byte()
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def get_mask(self, size, times = 7):
        # mask = torch.ones_like(data['image'])
        scale = 4
        b,_,x,y = size
        mask = torch.rand(b,1,x//scale,y//scale).cuda()
        pool = torch.nn.AvgPool2d(3, stride=1,padding=1)
        mask = (mask > 0.5).float()

        for i in range(times):
            mask = pool(mask)
            mask = (mask > 0.5).float()
        
        if scale > 1:
            mask = torch.nn.functional.interpolate(mask, size=(x,y))

        return 1 - mask

    def get_semantics(self, label_map):
        # create one-hot label map
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)
        return input_semantics
            

