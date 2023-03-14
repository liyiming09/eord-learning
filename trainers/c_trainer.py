"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel
from models.attentioneffect_model import AttentionEffectModel
from models.learneord_model import LearnEordModel
from models.cls_model import ClsModel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

class CTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt

        # local_rank = self.opt.local_rank

        # 新增3：DDP backend初始化
        #   a.根据local_rank来设定当前使用哪块GPU
        # torch.cuda.set_device(local_rank)
        # #   b.初始化DDP，使用默认backend(nccl)就行。如果是CPU模型运行，需要选择其他后端。
        # dist.init_process_group(backend='nccl')

        # 新增4：定义并把模型放置到单独的GPU上，需要在调用`model=DDP(model)`前做哦。
        #       如果要加载模型，也必须在这里做哦。
        # device = torch.device("cuda", local_rank)

        # self.model = Pix2PixModel(opt)
        self.model = ClsModel(opt)

        if len(opt.gpu_ids) > 0:
            # self.model = DataParallelWithCallback(self.model,device_ids=opt.gpu_ids)
            # # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(device)
            # # self.model = DDP(self.model,find_unused_parameters=False,  device_ids=[local_rank], output_device=local_rank)
            # self.model.cuda()
            # self.model_on_one_gpu = self.model.module


            self.model.cuda()
            self.model_on_one_gpu = self.model
            

            #--------------------
            # self.model = DataParallelWithCallback(self.model,
            #                                               device_ids=opt.gpu_ids)
            # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(device)
            # self.model = DDP(self.model,find_unused_parameters=True,  device_ids=[local_rank], output_device=local_rank)
            # self.model.cuda()
            # self.model_on_one_gpu = self.model
        else:
            self.model_on_one_gpu = self.model

        self.generated = None

        if opt.isTrain:
            self.optimizer_C =  self.model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

        # print(self.model_on_one_gpu.netG)
        # print(self.model_on_one_gpu.netD)



    def run_classifier_one_step(self,data):
        self.logs = {}
        self.optimizer_C.zero_grad()
        losses, masked, semantics, logs= self.model(data, mode='classifier')
        # print(654321)
        loss = sum(losses.values()).mean()
        loss.backward()
        self.optimizer_C.step()
        # 
        self.losses = losses
        self.semantics = semantics
        self.masked = masked
        self.losses['C_Accuracy'] = torch.Tensor([logs[0]/logs[1]])
        # self.logs['correct'] = logs[0]dd
        # self.logs['total'] = logs[1]

    def get_scoremap_one_step(self,data):
        maps, masked_image, semantics_seq = self.model(data, mode='backword')

        self.maps = maps

    # def run_discriminator_one_step(self, data):
    #     self.optimizer_E.zero_grad()
    #     self.optimizer_ED.zero_grad()
    #     d_losses = self.model(data, mode='interventor_discriminator')
    #     # print(654321)d
    #     d_loss = sum(d_losses.values()).mean()
    #     d_loss.backward()
    #     self.optimizer_ED.step()
    #     # self.optimizer_ED.zero_grad()
    #     self.d_losses = d_losses

    # def run_interventor_one_step(self, data):
    #     self.optimizer_E.zero_grad()
    #     with torch.autograd.set_detect_anomaly(True):
    #         g_losses, generated, masked, semantics = self.model(data, mode='interventor')
    #     # print(123456)
    #         g_loss = sum(g_losses.values()).mean()
        

    #         g_loss.backward()
    #     self.optimizer_E.step()
    #     # self.optimizer_E.zero_grad()
    #     self.g_losses = g_losses
    #     self.generated = generated
    #     self.masked = masked
    #     self.semantics = semantics
    #     self.d_losses = {}

    def get_latest_losses(self):
        return {**self.losses}

    # def get_input_image(self):
    #     return self.images
    def get_acc(self):
        return self.logs
    def get_maps(self):
        return self.maps

    def get_semantics(self):
        return self.semantics[0]

    def get_intervention(self):
        # base, pos, neg = self.semantics[1], self.semantics[6], self.semantics[7]
        return self.semantics[1]


    def get_mask(self):
        if self.masked.shape[1] == 3:
            return self.masked
        else:
            return self.masked[:,:3]

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.model_on_one_gpu.save(epoch)

    
    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        new_lr = self.opt.lr * 0.1**(epoch//15)

        if new_lr != self.old_lr:

            new_lr_C = new_lr


            for param_group in self.optimizer_C.param_groups:
                param_group['lr'] = new_lr_C

            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
