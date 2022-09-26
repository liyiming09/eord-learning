""".unsqueeze(0)
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torchvision.transforms as t 
import models.networks as networks
import util.util as util
from random import randint, random

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import numpy as np

class Pix2PixAModel(torch.nn.Module):

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

       
        

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_style_loss:
                self.criterionStyle = networks.StyleLoss(self.opt.gpu_ids)
            if self.opt.divco:
                self.criterionDivco = networks.DivcoLoss(self.opt)
                self.criterionModeseek = networks.ModeSeekingLoss(self.opt)

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

        if  self.opt.divco:
            tmp_semantics = []
            base_size = self.opt.batchSize//len(self.opt.gpu_ids)
            for k in range(input_semantics.shape[0]):
                tmp_semantics.append(input_semantics[k:k+1,[self.cls[k%base_size,:]]])
            self.invent_semantics = torch.cat(tmp_semantics, 0)
            # print(self.invent_semantics.shape)

        if mode == 'generator':
            # self.set_requires_grad(self.netD, False)
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, masked_image)
            return g_loss, generated, masked_image, input_semantics
        elif mode == 'discriminator':
            # self.set_requires_grad(self.netD, True)
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, masked_image)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, atts = self.generate_fake(input_semantics, real_image, masked_image)
            return fake_image, masked_image, input_semantics, atts
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())

        if opt.isTrain:
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)

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
        dist.init_process_group(backend='nccl')

        # 新增4：定义并把模型放置到单独的GPU上，需要在调用`model=DDP(model)`前做哦。
        #       如果要加载模型，也必须在这里做哦。
        device = torch.device("cuda", local_rank)
        if len(opt.gpu_ids) > 0:
            # self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
            #                                               device_ids=opt.gpu_ids)
            netG = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netG).to(device)
            netG = DDP(netG,find_unused_parameters=True,  device_ids=[local_rank], output_device=local_rank).cuda()
            if opt.isTrain:
                netD = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netD).to(device)
                netD = DDP(netD,find_unused_parameters=True,  device_ids=[local_rank], output_device=local_rank).cuda()

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netD

    # preprocess the input, such as moving the tensors to GPUs and`
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        self.cls = data['cls']
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
        if self.opt.eord:
            batchsize = label.shape[0]
            self.eord_flag = False
            zeromap = torch.zeros_like(label).cuda()
            for i in range(batchsize):
                # print(data['ped_label'][0].device, zeromap.device)
                flag = data['ped_label'][0].equal(zeromap)
               
                self.eord_flag = self.eord_flag or flag
        if self.opt.eord  and (not self.eord_flag):
            pos_label_list = data['ped_label']
            pos_labels = torch.cat(pos_label_list, 0).long().cuda()
            neg_label_list = data['ned_label']
            neg_labels = torch.cat(neg_label_list, 0).long().cuda()

                
        # Get Semantics
        input_semantics = self.get_semantics(label)
        if self.opt.eord  and (not self.eord_flag):
            extra_input_semantics = self.get_semantics(torch.cat((pos_labels,neg_labels),0))
            input_semantics = torch.cat((input_semantics, extra_input_semantics), dim=0)
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
            if self.opt.eord and (not self.eord_flag):
                mask = mask.repeat(self.opt.num_negative + 2,1,1,1)
                image = image.repeat(self.opt.num_negative + 2,1,1,1)
                masked = masked.repeat(self.opt.num_negative + 2,1,1,1)

            if self.opt.segmentation_mask:
                
                input_semantics *= (1-mask)


            input_semantics = torch.cat([input_semantics,1-mask[:,0:1]],dim=1)#input_semantics+遮挡区域

        # print(image.shape, image.device,'img')
        return input_semantics, image, masked

    def compute_generator_loss(self, input_semantics, real_image, masked_image):
        G_losses = {}

        fake_image = self.generate_fake(
            input_semantics, real_image, masked_image)

        if self.opt.divco and (not self.eord_flag):
            pred_fake, pred_real, feat = self.discriminate(
                input_semantics, fake_image, real_image, enc_feat = True)
        else:
            pred_fake, pred_real = self.discriminate(
                input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

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
        if self.opt.divco and (not self.eord_flag):
            G_losses['Divco'] = self.criterionDivco(feat) \
                * self.opt.lambda_divco
        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image, masked_image):
        D_losses = {}
        with torch.no_grad():
            fake_image = self.generate_fake(input_semantics, real_image, masked_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        # pred_fake, pred_real = self.discriminate(
        #     input_semantics, fake_image, real_image)
        if self.opt.divco and (not self.eord_flag):
            pred_fake, pred_real, feat = self.discriminate(
                input_semantics, fake_image, real_image, enc_feat = True)
        else:
            pred_fake, pred_real = self.discriminate(
                input_semantics, fake_image, real_image)

        mask = input_semantics[:,[-1]]
        # print(self.cls,self.cls.shape)
        # tmp_semantics = []
        # for k in range(input_semantics.shape[0]//len(self.opt.gpu_ids)):
        #     tmp_semantics.append(input_semantics[:,[self.cls[k,:]]])
        # invent_semantics = torch.cat(tmp_semantics, 1)
        # print(tmp_semantics[-1].shape,invent_semantics.shape)
        # invent_semantics = input_semantics[:,[self.cls]]
        # print(input_semantics.shape, invent_semantics.shape)
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)
        if self.opt.divco and (not self.eord_flag):
            D_losses['D_ms'] = self.criterionModeseek(feat,self.invent_semantics)* self.opt.lambda_ms#
        return D_losses

    def generate_fake(self, input_semantics, real_image, masked_image = None):
        if not self.opt.no_inpaint:
            mask = input_semantics[:,[-1]]
            fake_image, att = self.netG(masked_image, input_semantics, getatt = True)

            if not self.opt.no_mix_real_fake:
                fake_image = (1 - mask) * real_image + mask * fake_image 
        else:
            fake_image = self.netG(input_semantics, masked_image)


        return fake_image, att

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image, enc_feat = False):
        # if self.opt.netG == 'unet':
        #     input_semantics = input_semanno_tics[:,:,self.mask_x[0]:self.mask_x[1],\
        #                                           self.mask_y[0]:self.mask_y[1]]
        #     fake_image = fake_image[:,:,self.mask_x[0]:self.mask_x[1],\
        #                                           self.mask_y[0]:self.mask_y[1]]
        #     real_image = real_image[:,:,self.mask_x[0]:self.mask_x[1],\
        #                                           self.mask_y[0]:self.mask_y[1]]




        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        # print(fake_and_real.shape,000)

        if not enc_feat:
            discriminator_out = self.netD(fake_and_real)
            pred_fake, pred_real = self.divide_pred(discriminator_out)
            return pred_fake, pred_real
        else:
            discriminator_out, feat = self.netD(fake_and_real, enc_feat)
        
            pred_fake, pred_real = self.divide_pred(discriminator_out)
            # print(pred_fake[0][0].shape)
            return pred_fake, pred_real, feat

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                if self.opt.divco and (not self.eord_flag):
                    fake.append([tensor[:2] for tensor in p])
                    real.append([tensor[tensor.size(0) // 2:tensor.size(0) // 2 + 2] for tensor in p])
                else:
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
            

