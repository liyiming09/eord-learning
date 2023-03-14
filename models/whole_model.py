""".unsqueeze(0)
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from curses import A_ALTCHARSET
import torch
import torchvision.transforms as t 
import models.networks as networks
import util.util as util
from random import randint, random

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import numpy as np

class WholeModel(torch.nn.Module):

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

        

        self.netG, self.netD = self.initialize_networks(opt)
        if self.opt.monce or self.opt.patchnce or self.opt.masknce:
            self.netF = self.initialize_netF(opt)
        if self.opt.vae: self.netE = self.initialize_networksE(opt)

       
        

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterioncls = torch.nn.CrossEntropyLoss()
            self.criterionsimlar = networks.CosLoss()
            self.criterionRecons = torch.nn.MSELoss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_style_loss:
                self.criterionStyle = networks.StyleLoss(self.opt.gpu_ids)
            if self.opt.divco:
                self.criterionBYol = networks.ByolLoss(self.opt)
            if self.opt.modeseek:
                self.criterionModeseek = networks.ModeSeekingLoss(self.opt)
            if self.opt.monce:
                self.criterionMoNCE = networks.MoNCELoss(self.opt, self.netF)
            if self.opt.masknce:
                if self.opt.use_queue:
                    self.criterionMaskNCE = networks.MaskNCELoss_queue(self.opt, self.netF)
                else:
                    self.criterionMaskNCE = networks.MaskNCELoss(self.opt, self.netF)
            if self.opt.patchnce:
                self.criterionPatchNCE = networks.PatchLoss(self.opt, self.netF)
            if self.opt.effect:
                self.criterionEffect = networks.EffectLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)

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
        if 'online' in mode:
            input_semantics, real_image, image_mask = self.preprocess_input_online(data)
        else:
            input_semantics, real_image, image_mask = self.preprocess_input(data)


        self.invent_semantics = []
        self.base_size = self.opt.batchSize//len(self.opt.gpu_ids)
        for it in range(len(input_semantics)):
            tmp_semantics = []
            for k in range(input_semantics[it].shape[0]):
                tmp_semantics.append(input_semantics[it][k:k+1,[self.cls[k%self.base_size,:]]])
            self.invent_semantics.append(torch.cat(tmp_semantics,dim = 0))

        if mode == 'generator':
            # self.set_requires_grad(self.netD, False)
            # print(1)
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, image_mask)
            return g_loss, generated, image_mask, input_semantics + self.invent_semantics + [self.intervent_pos_mask, self.intervent_neg_mask]
        elif mode == 'generator_online':
            # self.set_requires_grad(self.netD, False)
            # print(1)
            g_loss, generated = self.compute_generator_loss_online(
                input_semantics, real_image, image_mask)
            return g_loss, generated, image_mask, input_semantics + self.invent_semantics + [self.intervent_pos_mask, self.intervent_neg_mask]
        elif mode == 'generator_noeord':
            g_loss, generated = self.compute_generator_loss_noeord(
                input_semantics, real_image, image_mask)
            return g_loss, generated, image_mask, input_semantics + self.invent_semantics
        elif mode == 'interventor':
            int_loss = self.compute_interventor_loss(input_semantics, real_image, image_mask)
            return int_loss, 0 , image_mask, input_semantics + self.invent_semantics + [self.intervent_pos_mask, self.intervent_neg_mask]
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(input_semantics, real_image, image_mask)
            return d_loss
        elif mode == 'discriminator_noeord':
            d_loss = self.compute_discriminator_loss_noeord(input_semantics, real_image, image_mask)
            return d_loss
        elif mode == 'discriminator_online':
            d_loss = self.compute_discriminator_loss_online(input_semantics, real_image, image_mask)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                fake_images = self.generate_fake( input_semantics[0], real_image, image_mask)
                fake_image, pos_image, neg_image = fake_images[0*self.base_size:1*self.base_size,...], fake_images[1*self.base_size:2*self.base_size,...], fake_images[2*self.base_size:3*self.base_size,...]

                return [fake_image, pos_image, neg_image], image_mask, input_semantics + self.invent_semantics + [self.intervent_pos_mask, self.intervent_neg_mask]
        elif mode == 'attention':
            with torch.no_grad():

                fake_images, atts = self.generate_fake( input_semantics[0], real_image, image_mask, mode = 'attention')
                fake_image, pos_image, neg_image = fake_images[0*self.base_size:1*self.base_size,...], fake_images[1*self.base_size:2*self.base_size,...], fake_images[2*self.base_size:3*self.base_size,...]
                fake_att, pos_att, neg_att = [att[0*self.base_size:1*self.base_size,...] for att in atts], [att[1*self.base_size:2*self.base_size,...] for att in atts], [ att[2*self.base_size:3*self.base_size,...] for att in atts]
                return [fake_image, pos_image, neg_image], image_mask, input_semantics + self.invent_semantics + [self.intervent_pos_mask, self.intervent_neg_mask], [fake_att, pos_att, neg_att]
        elif mode == 'attention_online':
            with torch.no_grad():
                fake_images, atts = self.generate_fake( input_semantics, real_image, image_mask, mode = 'attention_online')
                fake_image, pos_image, neg_image = fake_images[0*self.base_size:1*self.base_size,...], fake_images[1*self.base_size:2*self.base_size,...], fake_images[2*self.base_size:3*self.base_size,...]
                fake_att, pos_att, neg_att = [att[0*self.base_size:1*self.base_size,...] for att in atts], [att[1*self.base_size:2*self.base_size,...] for att in atts], [ att[2*self.base_size:3*self.base_size,...] for att in atts]
                return [fake_image, pos_image, neg_image], image_mask, input_semantics + self.invent_semantics + [self.intervent_pos_mask, self.intervent_neg_mask], [fake_att, pos_att, neg_att]
        elif mode == 'baseline_inference':
            with torch.no_grad():

                fake_image = self.generate_fake( input_semantics[0], real_image, image_mask, mode = 'noeord')
                return fake_image, image_mask, input_semantics + self.invent_semantics 
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        # E_params = list(self.netE.parameters())
        G_params = list(self.netG.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())
            # ED_params = list(self.netED.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr, ED_lr  = opt.lr, opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr, ED_lr = opt.lr / 2, opt.lr * 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr/10, betas=(beta1, beta2))
        # optimizer_E = torch.optim.Adam(E_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        # optimizer_ED = torch.optim.Adam(ED_params, lr=ED_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        # util.save_network(self.netE, 'E', epoch, self.opt)
        # util.save_network(self.netED, 'ED', epoch, self.opt)
        # if self.opt.monce:
        #     util.save_network(self.netF, 'F', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None

        local_rank = self.opt.local_rank

        # 新增3：DDP backend初始化
        #   a.根据local_rank来设定当前使用哪块GPU
        torch.cuda.set_device(local_rank)
        #   b.初始化DDP，使用默认backend(nccl)就行。如果是CPU模型运行，需要选择其他后端。
        # dist.init_process_group(backend='nccl')

        # 新增4：定义并把模型放置到单独的GPU上，需要在调用`model=DDP(model)`前做哦。
        #       如果要加载模型，也必须在这里做哦。
        device = torch.device("cuda", local_rank)
        if len(opt.gpu_ids) > 0:
            # self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
            #                                               device_ids=opt.gpu_ids)
            netG = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netG).to(device)
            netG = DDP(netG,find_unused_parameters=False,  device_ids=[local_rank], output_device=local_rank).cuda()
            if opt.isTrain:
                netD = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netD).to(device)
                netD = DDP(netD,find_unused_parameters=True,  device_ids=[local_rank], output_device=local_rank).cuda()


        
        # if opt.isTrain: 
        #     save_dir = '../attention-divco-projector/checkpoints/effect-spade-unet-10-1/'
        #     netG = util.load_pretrained_net(netG, 'G', 'latest', save_dir, opt)
        #     netD = util.load_pretrained_net(netD, 'D', 'latest', save_dir, opt)
        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netD
    
    def initialize_networksE(self, opt):
        netE = networks.define_E(opt)
        # netED = networks.define_ED(opt) if opt.isTrain else None

        local_rank = self.opt.local_rank
        device = torch.device("cuda", local_rank)
        if len(opt.gpu_ids) > 0:
            # self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
            #                                               device_ids=opt.gpu_ids)
            netE = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netE).to(device)
            netE = DDP(netE,find_unused_parameters=False,  device_ids=[local_rank], output_device=local_rank).cuda()
            # if opt.isTrain:
            #     netED = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netED).to(device)
            #     netED = DDP(netED,find_unused_parameters=False,  device_ids=[local_rank], output_device=local_rank).cuda()
            

        save_dir = '../eord-learning/checkpoints/only-E-4-vae-D/' if  'city' in self.opt.dataset_mode else '../eord-learning/checkpoints/onlyE-ade-1/'
        netE = util.load_pretrained_net(netE, 'E', 'latest', save_dir, opt)
        # if not opt.isTrain or opt.continue_train:
        #     netE = util.load_network(netE, 'E', opt.which_epoch, opt)
            # if opt.isTrain:
                # netED = util.load_network(netED, 'ED', opt.which_epoch, opt)
        return netE
        # return netE, netED
# 


    def initialize_netF(self, opt):
        netF = networks.define_F(opt) if opt.isTrain else None

        # if not opt.isTrain or opt.continue_train:
        #     if opt.isTrain:
        #         netF = util.load_network(netF, 'F', opt.which_epoch, opt)

        return  netF

    # preprocess the input, such as moving the tensors to GPUs and`
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        self.cls = data['cls']
        if self.opt.monce or self.opt.patchnce or self.opt.masknce:
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
        # self.eord_flag = False
        # # if self.opt.eord:
        # #     batchsize = label.shape[0]
            
        #     # zeromap = torch.zeros_like(label).cuda()
        #     # for i in range(batchsize):
        #     #     # print(data['ped_label'][0].device, zeromap.device)
        #     #     flag = data['ned_label'][0].equal(zeromap)
        #     #     print(flag)
        #     #     self.eord_flag = self.eord_flag or flag
        # if self.opt.eord  and (not self.eord_flag):
        #     pos_label_list = data['ped_label']
        #     pos_labels = torch.cat(pos_label_list, 0).long().cuda()
        #     neg_label_list = data['ned_label']
        #     neg_labels = torch.cat(neg_label_list, 0).long().cuda()

                
        # Get Semantics
        input_semantics = self.get_semantics(label)
        # if self.opt.eord  and (not self.eord_flag):
        #     pos_input_semantics = self.get_semantics(pos_labels)
        #     neg_input_semantics = self.get_semantics(neg_labels)
            # input_semantics = torch.cat((input_semantics, extra_input_semantics), dim=0)
        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = inst
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)


        if "mask" in data.keys():
            mask = data["mask"]
            if self.use_gpu():
                mask = mask.cuda()

        elif self.opt.bbox:
            mask = 1 - mask_in#背景区域
                
        else:
            # Cellural Mask used for AIM2020 Challenge on Image Extreme Inpainting
            mask = self.get_mask(image.size())

        #if erode or dilate, repeat the other tensor so as to match the dimension of input_semantics

        semantics = [input_semantics]
        # print(image.shape, image.device,'img')
        return semantics, image, mask

    def preprocess_input_online(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        self.cls = data['cls']
        if self.opt.monce or self.opt.patchnce or self.opt.masknce:
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
        # if self.opt.eord:
        #     batchsize = label.shape[0]
            
            # zeromap = torch.zeros_like(label).cuda()
            # for i in range(batchsize):
            #     # print(data['ped_label'][0].device, zeromap.device)
            #     flag = data['ned_label'][0].equal(zeromap)
            #     print(flag)
            #     self.eord_flag = self.eord_flag or flag
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


        semantics = [input_semantics, pos_input_semantics, neg_input_semantics]

        # print(image.shape, image.device,'img')
        return semantics, image, mask


    def compute_interventor_loss(self, input_semantics, real_image, image_mask):
        I_losses = {}
        self.generate_intervention(input_semantics[0], real_image, image_mask)

        # pred_fake, pred_real, pred_mask_base, pred_mask_real = self.discriminate(input_semantics[0], fake_image, real_image, mode = 'base', is_generate = True)
        # pred_fake_pos, pred_mask_pos = self.discriminate(self.intervent_pos_mask, pos_image, real_image, mode = 'pos', is_generate = True)
        # pred_fake_neg, pred_mask_neg = self.discriminate(self.intervent_neg_mask, neg_image, real_image, mode = 'neg', is_generate = True)
        #pred_mask包括两部分，cls的分类结果，和是否进行了干预，true代表无干预，false代表有

        I_losses['Mask_recons'] = (self.criterionRecons(self.intervent_pos_mask, self.invent_semantics[1]) + self.criterionRecons(self.intervent_neg_mask, self.invent_semantics[2]) ) * self.opt.lambda_recons
        # I_losses['Mask'] = self.criterionGAN(pred_mask_real, True,for_discriminator=False) \
        #     + self.criterionGAN(pred_mask_base, True,for_discriminator=False) \
        #         + self.criterionGAN(pred_mask_pos, True,for_discriminator=False) \
        #             + self.criterionGAN(pred_mask_neg, False,for_discriminator=False)



        return I_losses

    def compute_generator_loss(self, input_semantics, real_image, image_mask):
        G_losses = {}
        fake_images = self.generate_fake( input_semantics[0], real_image, image_mask)
        fake_image, pos_image, neg_image = fake_images[0*self.base_size:1*self.base_size,...], fake_images[1*self.base_size:2*self.base_size,...], fake_images[2*self.base_size:3*self.base_size,...]
        # pos_image, neg_image = self.generate_fake(input_semantics[0], real_image, image_mask, mode = 'intervention')
        # print(2)
        # if (self.opt.divco) and (not self.eord_flag):
        #     pred_fake, pred_real, pred_neg, feat = self.discriminate(
        #         input_semantics, fake_image, real_image, enc_feat = True)
        # elif self.opt.effect:
        #     pred_fake, pred_real, pred_neg = self.discriminate(
        #         input_semantics, fake_image, real_image)
        # else:
        pred_fake, pred_real = self.discriminate(input_semantics[0], fake_image, real_image, mode = 'base', is_generate = True)
        pred_fake_pos = self.discriminate(self.whole_pos_mask, pos_image, real_image, mode = 'pos', is_generate = True)
        pred_fake_neg = self.discriminate(self.whole_neg_mask, neg_image, real_image, mode = 'neg', is_generate = True)
        #pred_mask包括两部分，cls的分类结果，和是否进行了干预，true代表无干预，false代表有

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,for_discriminator=False) \
            + self.criterionGAN(pred_fake_pos, True,for_discriminator=False) 
        # G_losses['Mask_recons'] = (self.criterionRecons(self.intervent_pos_mask, self.invent_semantics[1]) + self.criterionRecons(self.intervent_neg_mask, self.invent_semantics[2]) ) * self.opt.lambda_recons
        # G_losses['Mask'] = self.criterionGAN(pred_mask_real, True,for_discriminator=False) \
        #     + self.criterionGAN(pred_mask_base, True,for_discriminator=False) \
        #         + self.criterionGAN(pred_mask_pos, True,for_discriminator=False) \
        #             + self.criterionGAN(pred_mask_neg, False,for_discriminator=False)
            #         self.criterioncls(pred_mask_base[0], self.cls.squeeze(1)) \
            # + self.criterioncls(pred_mask_pos[0], self.cls.squeeze(1)) +  self.criterioncls(pred_mask_neg[0], self.cls.squeeze(1))
        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        if self.opt.use_style_loss:
            G_losses['Style'] = self.criterionStyle(fake_image, real_image) \
                * self.opt.lambda_style

        if self.opt.effect:
            #效应计算公式：base-neg = (干预前后差别区域，real)
            # print(pred_neg[0][0].shape)
            # pred_effect = -pred_neg[0][0] + pred_fake[0][0][0:1,...]
            #----------------------1. neg-base
            # effect_map = self.intervent_neg_mask - self.invent_semantics[0]
            # effect_map[effect_map!=0] = 1
            #----------------------2. soft-mask: 1-neg
            src_effect_map = self.intervent_neg_mask.clone()
            src_effect_map[src_effect_map!=0] = 1#赋值为float32类型
            effect_map = src_effect_map - self.intervent_neg_mask

            G_losses['Effect'] = self.criterionEffect(pred_fake_neg, pred_fake,effect_map,  True, for_discriminator=False)* self.opt.lambda_effect
        if self.opt.recons_loss:
                G_losses['Recons'] = self.criterionRecons(fake_image, real_image) * self.opt.lambda_recons
        if self.opt.monce:
            G_losses['MoNCE'] = self.criterionMoNCE(fake_image, real_image, self.bbox)* self.opt.lambda_monce
        
        if self.opt.masknce:
            neg_map_threshold = self.intervent_neg_mask.clone()
            neg_map_threshold[neg_map_threshold < 0.2] = 0
            masknce_map = neg_map_threshold - self.invent_semantics[0]
            masknce_map[masknce_map!=0] = 1
            G_losses['MaskNCE'] = self.criterionMaskNCE(fake_image, real_image, self.bbox, mask_pos = self.intervent_pos_mask, mask_neg = masknce_map,  cur_cls = self.cls)* self.opt.lambda_masknce

        if self.opt.patchnce:
            G_losses['PatchNCE'] = self.criterionPatchNCE(fake_image, real_image, self.bbox)* self.opt.lambda_monce

        return G_losses, fake_image
    
    def compute_generator_loss_online(self, input_semantics, real_image, image_mask):
        G_losses = {}
        fake_images = self.generate_fake( input_semantics, real_image, image_mask, mode='online')
        fake_image, pos_image, neg_image = fake_images[0*self.base_size:1*self.base_size,...], fake_images[1*self.base_size:2*self.base_size,...], fake_images[2*self.base_size:3*self.base_size,...]
        # pos_image, neg_image = self.generate_fake(input_semantics[0], real_image, image_mask, mode = 'intervention')
        # print(2)
        # if (self.opt.divco) and (not self.eord_flag):
        #     pred_fake, pred_real, pred_neg, feat = self.discriminate(
        #         input_semantics, fake_image, real_image, enc_feat = True)
        # elif self.opt.effect:
        #     pred_fake, pred_real, pred_neg = self.discriminate(
        #         input_semantics, fake_image, real_image)
        # else:
        pred_fake, pred_real = self.discriminate(input_semantics[0], fake_image, real_image, mode = 'base', is_generate = True)
        pred_fake_pos = self.discriminate(input_semantics[1], pos_image, real_image, mode = 'pos', is_generate = True)
        pred_fake_neg = self.discriminate(input_semantics[2], neg_image, real_image, mode = 'neg', is_generate = True)
        #pred_mask包括两部分，cls的分类结果，和是否进行了干预，true代表无干预，false代表有

        G_losses['GAN_fake'] = self.criterionGAN(pred_fake, True,for_discriminator=False)
        G_losses['GAN_pos'] = self.criterionGAN(pred_fake_pos, True,for_discriminator=False) 
        # G_losses['Mask_recons'] = (self.criterionRecons(self.intervent_pos_mask, self.invent_semantics[1]) + self.criterionRecons(self.intervent_neg_mask, self.invent_semantics[2]) ) * self.opt.lambda_recons
        # G_losses['Mask'] = self.criterionGAN(pred_mask_real, True,for_discriminator=False) \
        #     + self.criterionGAN(pred_mask_base, True,for_discriminator=False) \
        #         + self.criterionGAN(pred_mask_pos, True,for_discriminator=False) \
        #             + self.criterionGAN(pred_mask_neg, False,for_discriminator=False)
            #         self.criterioncls(pred_mask_base[0], self.cls.squeeze(1)) \
            # + self.criterioncls(pred_mask_pos[0], self.cls.squeeze(1)) +  self.criterioncls(pred_mask_neg[0], self.cls.squeeze(1))
        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        if self.opt.use_style_loss:
            G_losses['Style'] = self.criterionStyle(fake_image, real_image) \
                * self.opt.lambda_style

        if self.opt.effect:
            #效应计算公式：base-neg = (干预前后差别区域，real)
            # print(pred_neg[0][0].shape)
            # pred_effect = -pred_neg[0][0] + pred_fake[0][0][0:1,...]
            #----------------------1. neg-base
            # effect_map = self.intervent_neg_mask - self.invent_semantics[0]
            # effect_map[effect_map!=0] = 1
            #----------------------2. soft-mask: 1-neg
            src_effect_map = self.intervent_neg_mask.clone()
            src_effect_map[src_effect_map!=0] = 1#赋值为float32类型
            effect_map = src_effect_map - self.intervent_neg_mask

            G_losses['Effect'] = self.criterionEffect(pred_fake_neg, pred_fake,effect_map,  True, for_discriminator=False)* self.opt.lambda_effect
        if self.opt.recons_loss:
                G_losses['Recons'] = self.criterionRecons(fake_image, real_image) * self.opt.lambda_recons
        if self.opt.monce:
            G_losses['MoNCE'] = self.criterionMoNCE(fake_image, real_image, self.bbox)* self.opt.lambda_monce
        
        if self.opt.masknce:
            neg_map_threshold = self.intervent_neg_mask.clone()
            neg_map_threshold[neg_map_threshold < 0.2] = 0
            masknce_map = neg_map_threshold - self.invent_semantics[0]
            masknce_map[masknce_map!=0] = 1
            G_losses['MaskNCE'] = self.criterionMaskNCE(fake_image, real_image, self.bbox, mask_pos = self.intervent_pos_mask, mask_neg = masknce_map,  cur_cls = self.cls)* self.opt.lambda_masknce

        if self.opt.patchnce:
            G_losses['PatchNCE'] = self.criterionPatchNCE(fake_image, real_image, self.bbox)* self.opt.lambda_monce

        return G_losses, fake_image
    
    def compute_generator_loss_noeord(self, input_semantics, real_image, image_mask):
        G_losses = {}
        fake_image = self.generate_fake( input_semantics[0], real_image, image_mask, mode = 'noeord')

        pred_fake, pred_real = self.discriminate(input_semantics[0], fake_image, real_image, mode = 'base', is_generate = True)

        #pred_mask包括两部分，cls的分类结果，和是否进行了干预，true代表无干预，false代表有

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,for_discriminator=False) \

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        if self.opt.use_style_loss:
            G_losses['Style'] = self.criterionStyle(fake_image, real_image) \
                * self.opt.lambda_style

        if self.opt.recons_loss:
                G_losses['Recons'] = self.criterionRecons(fake_image, real_image) * self.opt.lambda_recons
        if self.opt.monce:
            G_losses['MoNCE'] = self.criterionMoNCE(fake_image, real_image, self.bbox)* self.opt.lambda_monce
        
        if self.opt.patchnce:
            G_losses['PatchNCE'] = self.criterionPatchNCE(fake_image, real_image, self.bbox)* self.opt.lambda_monce

        return G_losses, fake_image



    def compute_discriminator_loss(self, input_semantics, real_image, image_mask):
        D_losses = {}
        with torch.no_grad():
            fake_images = self.generate_fake(input_semantics[0], real_image, image_mask)
            fake_images = fake_images.detach()
            fake_images.requires_grad_()

            fake_image, pos_image, neg_image = fake_images[0*self.base_size:1*self.base_size,...], fake_images[1*self.base_size:2*self.base_size,...], fake_images[2*self.base_size:3*self.base_size,...]


        pred_fake, pred_real = self.discriminate(input_semantics[0], fake_image, real_image, mode = 'base')
        pred_fake_pos = self.discriminate(self.whole_pos_mask, pos_image, real_image, mode = 'pos')


        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,for_discriminator=True)
        D_losses['D_Pos'] = self.criterionGAN(pred_fake_pos, False,for_discriminator=True) 

        D_losses['D_real'] = self.criterionGAN(pred_real, True,for_discriminator=True)

        return D_losses

    def compute_discriminator_loss_online(self, input_semantics, real_image, image_mask):
        D_losses = {}
        with torch.no_grad():
            fake_images = self.generate_fake(input_semantics, real_image, image_mask, mode = 'online')
            fake_images = fake_images.detach()
            fake_images.requires_grad_()

            fake_image, pos_image, neg_image = fake_images[0*self.base_size:1*self.base_size,...], fake_images[1*self.base_size:2*self.base_size,...], fake_images[2*self.base_size:3*self.base_size,...]


        pred_fake, pred_real = self.discriminate(input_semantics[0], fake_image, real_image, mode = 'base')
        pred_fake_pos = self.discriminate(input_semantics[1], pos_image, real_image, mode = 'pos')


        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,for_discriminator=True)
        D_losses['D_Pos'] = self.criterionGAN(pred_fake_pos, False,for_discriminator=True) 

        D_losses['D_real'] = self.criterionGAN(pred_real, True,for_discriminator=True)

        return D_losses

    def compute_discriminator_loss_noeord(self, input_semantics, real_image, image_mask):
        D_losses = {}
        with torch.no_grad():
            fake_images = self.generate_fake(input_semantics[0], real_image, image_mask, mode = 'noeord')
            fake_image = fake_images.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(input_semantics[0], fake_image, real_image, mode = 'base')

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,for_discriminator=True)
        return D_losses

    def generate_intervention(self, input_semantics, real_image, image_mask = None, mode = 'base'):
        intervent_fakemask = self.netE(input_semantics, self.invent_semantics[0], mode = mode)
        self.intervent_pos_mask = intervent_fakemask[:,0:1,...].clone()
        self.intervent_neg_mask = intervent_fakemask[:,1:2,...].clone()

    def generate_fake(self, input_semantics, real_image, image_mask = None, mode = 'base'):
        #这一种实现方案是并行一次生成的

        mask = 1-image_mask
        if mode == 'base':
            with torch.no_grad():
                intervent_fakemask = self.netE(mask *input_semantics,  mask *self.invent_semantics[0], mode = mode)
                self.intervent_pos_mask = (1 - mask) * self.invent_semantics[0] + mask * intervent_fakemask[:,0:1,...].detach()
                self.intervent_neg_mask = (1 - mask) * self.invent_semantics[0] + mask * intervent_fakemask[:,1:2,...].detach()
            self.whole_pos_mask = input_semantics.clone()
            self.whole_neg_mask = input_semantics.clone()
            for k in range(input_semantics.shape[0]):
                self.whole_pos_mask[k:k+1,[self.cls[k%self.base_size,:]]] = self.intervent_pos_mask[k:k+1,...].detach()
                self.whole_neg_mask[k:k+1,[self.cls[k%self.base_size,:]]] = self.intervent_neg_mask[k:k+1,...].detach()
            eord_input_semantics = torch.cat((input_semantics,self.whole_pos_mask, self.whole_neg_mask ),0)

            fake_image = self.netG(eord_input_semantics)

            return fake_image
            
        elif mode == 'online':

            eord_input_semantics = torch.cat(input_semantics,0)
            fake_image = self.netG( eord_input_semantics)
            self.intervent_pos_mask = self.invent_semantics[1]
            self.intervent_neg_mask = self.invent_semantics[2]
            return fake_image

        elif mode == 'attention':
            # fake_image, atts = self.netG(image_mask, input_semantics, getatt = True)
            with torch.no_grad():
                intervent_fakemask = self.netE(mask *input_semantics,  mask *self.invent_semantics[0], mode = mode)
                self.intervent_pos_mask = (1 - mask) * self.invent_semantics[0] + mask * intervent_fakemask[:,0:1,...].detach()
                self.intervent_neg_mask = (1 - mask) * self.invent_semantics[0] + mask * intervent_fakemask[:,1:2,...].detach()
            self.whole_pos_mask = input_semantics.clone()
            self.whole_neg_mask = input_semantics.clone()
            for k in range(input_semantics.shape[0]):
                self.whole_pos_mask[k:k+1,[self.cls[k%self.base_size,:]]] = self.intervent_pos_mask[k:k+1,...].detach()
                self.whole_neg_mask[k:k+1,[self.cls[k%self.base_size,:]]] = self.intervent_neg_mask[k:k+1,...].detach()
            eord_input_semantics = torch.cat((input_semantics,self.whole_pos_mask, self.whole_neg_mask ),0)

            fake_image, atts = self.netG(eord_input_semantics, getatt = True)

            return fake_image, atts
        elif mode == 'attention_online':
            # fake_image, atts = self.netG(image_mask, input_semantics, getatt = True)

            eord_input_semantics = torch.cat(input_semantics,0)
            fake_image, atts = self.netG(eord_input_semantics, getatt = True)
            self.intervent_pos_mask = self.invent_semantics[1]
            self.intervent_neg_mask = self.invent_semantics[2]

            return fake_image, atts
            

        elif mode == 'noeord':
            fake_image = self.netG(input_semantics)

            return fake_image
        else:
            print("error in intervention generation stage!")



        return fake_image

    def generate_fake_inseries(self, input_semantics, real_image, image_mask = None, mode = 'base'):
        #这一种实现方案是串行分两次进行生成的
        if not self.opt.no_inpaint:
            mask = input_semantics[:,[-1]]
            if mode == 'base':
                fake_image = self.netG(image_mask, input_semantics)

                if not self.opt.no_mix_real_fake:
                    fake_image = (1 - mask) * real_image + mask * fake_image 
                return fake_image
            elif mode == 'test':
                fake_image, atts = self.netG(image_mask, input_semantics, getatt = True)

                if not self.opt.no_mix_real_fake:
                    fake_image = (1 - mask) * real_image + mask * fake_image 
                return fake_image, atts
            elif mode == 'intervention':
                intervent_fakemask = self.netE(input_semantics, mode = mode)
                self.intervent_pos_mask = intervent_fakemask[:,0:1,...].clone()
                self.whole_pos_mask = input_semantics.clone()
                for k in range(input_semantics.shape[0]):
                    self.whole_pos_mask[k:k+1,[self.cls[k%self.base_size,:]]] = self.intervent_pos_mask[k:k+1,...].clone()
                fake_pos_image = self.netG(image_mask, self.whole_pos_mask)

                self.intervent_neg_mask = intervent_fakemask[:,1:2,...].clone()
                self.whole_neg_mask = input_semantics.clone()
                for k in range(input_semantics.shape[0]):
                    self.whole_neg_mask[k:k+1,[self.cls[k%self.base_size,:]]] = self.intervent_neg_mask[k:k+1,...].clone()
                fake_neg_image = self.netG(image_mask, self.whole_neg_mask)

                if not self.opt.no_mix_real_fake:
                    fake_pos_image = (1 - mask) * real_image + mask * fake_pos_image 
                    fake_neg_image = (1 - mask) * real_image + mask * fake_neg_image 
                return fake_pos_image, fake_neg_image
            else:
                print("error in intervention generation stage!")
        else:
            fake_image = self.netG(input_semantics, image_mask)


        return fake_image

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image, enc_feat = False, mode = 'base', is_generate = False):
        # if self.opt.netG == 'unet':
        #     input_semantics = input_semanno_tics[:,:,self.mask_x[0]:self.mask_x[1],\
        #                                           self.mask_y[0]:self.mask_y[1]]
        #     fake_image = fake_image[:,:,self.mask_x[0]:self.mask_x[1],\
        #                                           self.mask_y[0]:self.mask_y[1]]
        #     real_image = real_image[:,:,self.mask_x[0]:self.mask_x[1],\
        #                                           self.mask_y[0]:self.mask_y[1]]



        if mode == 'base':
            fake_concat = torch.cat([input_semantics, fake_image], dim=1)
            real_concat = torch.cat([input_semantics, real_image], dim=1)

            # In Batch Normalization, the fake and real images are
            # recommended to be in the same batch to avoid disparate
            # statistics in fake and real images.
            # So both fake and real images are fed to D all at once.
            fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
            # print(fake_and_real.shape,000)


            discriminator_out = self.netD(fake_and_real)
            pred_fake, pred_real = self.divide_pred(discriminator_out)
            return pred_fake, pred_real
        elif mode == 'onlye_base':
            fake_concat = torch.cat([input_semantics, fake_image], dim=1)
            real_concat = torch.cat([input_semantics, real_image], dim=1)

            # In Batch Normalization, the fake and real images are
            # recommended to be in the same batch to avoid disparate
            # statistics in fake and real images.
            # So both fake and real images are fed to D all at once.
            fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
            # print(fake_and_real.shape,000)


            discriminator_out = self.netD(fake_and_real)
            pred_fake, pred_real = self.divide_pred(discriminator_out)
            pred_mask_fake = self.netED(torch.cat([self.invent_semantics[0], fake_image], dim=1), is_generate)
            pred_mask_real = self.netED(torch.cat([self.invent_semantics[0], real_image], dim=1), is_generate)
            return pred_fake, pred_real, pred_mask_fake, pred_mask_real

        else:#干预时，输入的mask只为1,H，W
            if mode == 'pos':
                fake_concat = torch.cat([input_semantics, fake_image], dim=1)
                # ED_cooncat = torch.cat([self.intervent_pos_mask, fake_image], dim=1)
            else:
                fake_concat = torch.cat([input_semantics, fake_image], dim=1)
                # ED_cooncat = torch.cat([self.intervent_neg_mask, fake_image], dim=1)

            if not enc_feat:
                pred_fake = self.netD(fake_concat)
                # pred_mask = self.netED(ED_cooncat, is_generate)
                return pred_fake
            else:
                pred_fake,feat = self.netD(fake_concat, enc_feat)
                # pred_mask = self.netED(ED_cooncat, is_generate)
                return pred_fake, feat

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        # print(4,pred[0][0].shape,pred[0][2].shape)

        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

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
            

