"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from data.basebox_dataset import BaseDataset as BaseNipsDataset
from data.baseconfounder_dataset import BaseDataset as BaseConfDataset

def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported. 
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and (issubclass(cls, BaseDataset) or issubclass(cls, BaseNipsDataset) or issubclass(cls, BaseConfDataset)):
            dataset = cls
            
    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):    
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt)
    opt.total_num = len(instance)
    print("dataset [%s] of size %d was created" %
          (type(instance).__name__, len(instance)))
    return instance

def create_dataloader(opt, instance):
     #新增1：使用DistributedSampler，DDP帮我们把细节都封装起来了。用，就完事儿！
    #       sampler的原理，后面也会介绍。
    train_sampler = torch.utils.data.distributed.DistributedSampler(instance)
    # 需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batchSize//len(opt.gpu_ids),
        shuffle=False,
        sampler=train_sampler,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain
    )
    return dataloader

def create_dataloader_noddp(opt, instance):
     #新增1：使用DistributedSampler，DDP帮我们把细节都封装起来了。用，就完事儿！
    #       sampler的原理，后面也会介绍。
    # train_sampler = torch.utils.data.distributed.DistributedSampler(instance)
    # 需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batchSize,
        shuffle=False,
        # sampler=train_sampler,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain
    )
    return dataloader
