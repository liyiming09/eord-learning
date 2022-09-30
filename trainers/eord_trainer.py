"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel
from models.attentioneffect_model import AttentionEffectModel
from models.learneord_model import LearningEordModel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

class EorDTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt

        local_rank = self.opt.local_rank

        # 新增3：DDP backend初始化
        #   a.根据local_rank来设定当前使用哪块GPU
        # torch.cuda.set_device(local_rank)
        # #   b.初始化DDP，使用默认backend(nccl)就行。如果是CPU模型运行，需要选择其他后端。
        # dist.init_process_group(backend='nccl')

        # 新增4：定义并把模型放置到单独的GPU上，需要在调用`model=DDP(model)`前做哦。
        #       如果要加载模型，也必须在这里做哦。
        device = torch.device("cuda", local_rank)

        # self.pix2pix_model = Pix2PixModel(opt)
        self.pix2pix_model = LearningEordModel(opt)

        if len(opt.gpu_ids) > 0:
            # self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
            #                                               device_ids=opt.gpu_ids)
            self.pix2pix_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.pix2pix_model).to(device)
            self.pix2pix_model = DDP(self.pix2pix_model,find_unused_parameters=True,  device_ids=[local_rank], output_device=local_rank)
            self.pix2pix_model.cuda()
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module

            #--------------------
            # self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
            #                                               device_ids=opt.gpu_ids)
            # self.pix2pix_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.pix2pix_model).to(device)
            # self.pix2pix_model = DDP(self.pix2pix_model,find_unused_parameters=True,  device_ids=[local_rank], output_device=local_rank)
            # self.pix2pix_model.cuda()
            # self.pix2pix_model_on_one_gpu = self.pix2pix_model
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None

        if opt.isTrain:
            self.optimizer_E, self.optimizer_D, self.optimizer_ED = \
                self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

        # print(self.pix2pix_model_on_one_gpu.netG)
        # print(self.pix2pix_model_on_one_gpu.netD)

    def run_generator_one_step(self, data):
        self.optimizer_E.zero_grad()
        g_losses, generated, masked, semantics = self.pix2pix_model(data, mode='generator')
        # print(123456)
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_E.step()
        self.optimizer_E.zero_grad()
        self.g_losses = g_losses
        self.generated = generated
        self.masked = masked
        self.semantics = semantics

    # def run_discriminator_one_step(self, data):
    #     self.optimizer_D.zero_grad()
    #     d_losses = self.pix2pix_model(data, mode='discriminator')
    #     d_loss = sum(d_losses.values()).mean()
    #     d_loss.backward()

    def run_discriminator_one_step(self, data):
        self.optimizer_ED.zero_grad()
        self.optimizer_D.zero_grad()
        d_losses = self.pix2pix_model(data, mode='discriminator')
        # print(654321)
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_ED.step()
        self.optimizer_D.step()
        self.optimizer_ED.zero_grad()
        self.optimizer_D.zero_grad()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def get_latest_real(self):
        return self.pix2pix_model_on_one_gpu.real_shape

    def get_semantics(self):
        return self.semantics

    def get_mask(self):
        if self.masked.shape[1] == 3:
            return self.masked
        else:
            return self.masked[:,:3]

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    
    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_E.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
