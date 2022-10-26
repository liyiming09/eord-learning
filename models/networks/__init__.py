"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from models.networks.base_network import BaseNetwork
from models.networks.loss import *
from models.networks.discriminator import *
from models.networks.generator import * 
from models.networks.sampler import * 
import util.util as util


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()

    netG_cls = find_network_using_name(opt.netG, 'generator')
    parser = netG_cls.modify_commandline_options(parser, is_train)
    if is_train:
        netD_cls = find_network_using_name(opt.netD, 'discriminator')
        parser = netD_cls.modify_commandline_options(parser, is_train)

        netE_cls = find_network_using_name(opt.netE, 'interventor')
        parser = netE_cls.modify_commandline_options(parser, is_train)

    return parser


def create_network(cls, opt, input_nc = None):
    if input_nc is None:
        net = cls(opt)
    else:
        net = cls(opt, input_nc)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)
    return net

def create_netF(cls, opt):

    net = cls(opt)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    net.init_weights('xaivier', 0.02)
    return net

def define_G(opt):
    netG_cls = find_network_using_name(opt.netG, 'generator')
    return create_network(netG_cls, opt)


def define_D(opt, input_nc=None):
    netD_cls = find_network_using_name(opt.netD, 'discriminator')
    return create_network(netD_cls, opt, input_nc)

def define_F(opt, input_nc=None):
    netF_cls = find_network_using_name(opt.netF + 'SampleF','sampler')
    return create_netF(netF_cls, opt)


def define_E(opt, input_nc=None):
    netE_cls = find_network_using_name(opt.netE ,'interventor')
    return create_network(netE_cls, opt)

def define_ED(opt, input_nc=None):
    netED_cls = find_network_using_name('LearnEorD', 'discriminator')
    return create_network(netED_cls, opt, input_nc)