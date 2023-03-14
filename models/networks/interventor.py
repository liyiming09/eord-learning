""" Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from email.policy import default
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.networks.architecture import SPADEEordBlock as SPADEEordBlock





class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=False)

    # def forward(self, g, x, fakeatt = False):
    #     g1 = self.W_g(g)
    #     x1 = self.W_x(x)
    #     psi = self.relu(g1 + x1)
    #     psi = self.psi(psi)
    #     if fakeatt:
    #         fake_att = torch.zeros_like(psi[0:1,...]).uniform_(0, 1)
    #         bad_att = psi[2:,...].clone()
    #         cf_att = psi.clone()
    #         cf_att[0:2,...] = psi[0:2,...] - fake_att
    #         cf_att[2:,...] = psi[2:,...] - fake_att
    #     out = x * psi
    #     return out
    def forward(self, g, x, fakeatt = False, getatt = False):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        if fakeatt:
            # fake_att = torch.zeros_like(psi[0:1,...]).uniform_(0, 1)
            bad_att = psi[2:,...].clone()
            cf_att = psi.clone()
            cf_att[0:2,...] = 2*psi[0:2,...] - bad_att
            # cf_att[2:,...] = psi[2:,...] - fake_att
            out = x * cf_att
            if getatt:
                return out, cf_att
            else:
                return out
        else:
            out = x * psi
            if getatt:
                return out, psi
            else:
                return out




class attentionunetInterventor(BaseNetwork):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self,opt):
        """Construct a Unet Interventor
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(attentionunetInterventor, self).__init__()

        self.opt = opt
        label_nc = opt.label_nc
        self.fakeatt = False
        input_nc = label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)+ (0 if opt.no_inpaint else 1)
        # if opt.mix_input_gen:
        #     input_nc += 4
        output_nc = 2
        # num_downs = 6
        # ngf=64

        # img_ch=3, output_ch=1

        ngf = 64
        filters = [ngf, ngf * 2, ngf * 4, ngf * 8, ngf * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.Conv1 = conv_block(input_nc, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])


        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_nc, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Sigmoid()


    def forward(self, x, extra = None, getatt = False, mode = 'base'):
        if getatt:
            # if self.opt.mix_input_gen:
            #     x = torch.cat([x,extra], dim = 1)
            atts= []
            e1 = self.Conv1(x)

            e2 = self.Maxpool1(e1)
            e2 = self.Conv2(e2)

            e3 = self.Maxpool2(e2)
            e3 = self.Conv3(e3)



            #print(x5.shape)
            d3 = self.Up3(e3)
            #print(d5.shape)

            x2, att3 = self.Att3(g=d3, x=e2, fakeatt = self.fakeatt, getatt = getatt)
            atts.append(att3)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            x1, att2 = self.Att2(g=d2, x=e1, fakeatt = self.fakeatt, getatt = getatt)
            atts.append(att2)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)

            out = self.Conv(d2)

            out = self.active(out)


            return out, atts
        else:


            e1 = self.Conv1(x)
            print(e1.get_device(),x.get_device())
            e2 = self.Maxpool1(e1)
            e2 = self.Conv2(e2)

            e3 = self.Maxpool2(e2)
            e3 = self.Conv3(e3)

            d3 = self.Up3(e3)
            #print(d5.shape)

            x2 = self.Att3(g=d3, x=e2, fakeatt = self.fakeatt)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            x1 = self.Att2(g=d2, x=e1, fakeatt = self.fakeatt)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)

            out = self.Conv(d2)

            out = self.active(out)

        return out


class vaeInterventor(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--use_vae', type = bool, default = True, help = "use a noise vector to generate the intervention map")
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super(vaeInterventor, self).__init__()
        self.opt = opt
        nf = opt.ngf

        self.opt = opt
        label_nc = opt.label_nc
        self.fakeatt = False
        input_nc = label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)+ (0 if opt.no_inpaint else 1)
        print("vae input_nc:",input_nc)
        # if opt.mix_input_gen:
        #     input_nc += 4
        output_nc = 2

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 8 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 8 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(8 * nf + 1, 8 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(8 * nf, 8 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(8 * nf, 8 * nf, opt)

        self.up_0 = SPADEResnetBlock(8 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, output_nc, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

        self.active = torch.nn.Sigmoid()

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.fineSize // (2**num_up_layers)
        sh = sw

        return sw, sh

    def forward(self, input, stuff,  z=None, mode = 'base'):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32).cuda()
            x = self.fc(z)
            # print(x.get_device(),z.get_device())
            x = x.view(-1, 8 * self.opt.ngf, self.sh, self.sw)
            cur_stuff = F.interpolate(stuff, size=(self.sh, self.sw))
            x = torch.cat((cur_stuff, x), dim = 1)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = self.active(x)

        return x

class spadeunetInterventor(BaseNetwork):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self,opt):
        """Construct a Unet Interventor
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(spadeunetInterventor, self).__init__()

        self.opt = opt
        label_nc = opt.label_nc
        self.fakeatt = self.opt.fakeattention
        input_nc = label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
        if opt.mix_input_gen:
            input_nc += 4
        output_nc = 3
        # num_downs = 6
        # ngf=64

        # img_ch=3, output_ch=1

        ngf = 64
        filters = [ngf, ngf * 2, ngf * 4, ngf * 8, ngf * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = SPADEResnetBlock(input_nc, filters[0], self.opt)
        self.Conv2 = SPADEResnetBlock(filters[0], filters[1], self.opt)
        self.Conv3 = SPADEResnetBlock(filters[1], filters[2], self.opt)
        self.Conv4 = SPADEResnetBlock(filters[2], filters[3], self.opt)
        self.Conv5 = SPADEResnetBlock(filters[3], filters[4], self.opt)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_nc, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x, extra = None, getatt = False):

        if self.opt.mix_input_gen:
            x = torch.cat([x,extra], dim = 1)

        e1 = self.Conv1(x, extra)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2, extra)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3, extra)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4, extra)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5, extra)

        #print(x5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
        if getatt:
            atts= []
            x4, att5 = self.Att5(g=d5, x=e4, fakeatt = self.fakeatt, getatt = getatt)
            atts.append(att5)
            d5 = torch.cat((x4, d5), dim=1)
            d5 = self.Up_conv5(d5)

            d4 = self.Up4(d5)
            x3, att4 = self.Att4(g=d4, x=e3, fakeatt = self.fakeatt, getatt = getatt)
            atts.append(att4)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            x2, att3 = self.Att3(g=d3, x=e2, fakeatt = self.fakeatt, getatt = getatt)
            atts.append(att3)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            x1, att2 = self.Att2(g=d2, x=e1, fakeatt = self.fakeatt, getatt = getatt)
            atts.append(att2)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)

            out = self.Conv(d2)

        #  out = self.active(out)


            return out, atts
        else:
            x4 = self.Att5(g=d5, x=e4, fakeatt = self.fakeatt)
            d5 = torch.cat((x4, d5), dim=1)
            d5 = self.Up_conv5(d5)

            d4 = self.Up4(d5)
            x3 = self.Att4(g=d4, x=e3, fakeatt = self.fakeatt)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            x2 = self.Att3(g=d3, x=e2, fakeatt = self.fakeatt)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            x1 = self.Att2(g=d2, x=e1, fakeatt = self.fakeatt)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2)

            out = self.Conv(d2)

        #  out = self.active(out)

            return out

class spadeunet2Interventor(BaseNetwork):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self,opt):
        #Construct a Unet Interventor

        super(spadeunet2Interventor, self).__init__()

        self.opt = opt
        label_nc = opt.label_nc
        self.fakeatt = False
        input_nc = label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1) + (0 if opt.no_inpaint else 1) 

        output_nc = 2
        print("input_nc:", input_nc)
        # num_downs = 6
        # ngf=64

        # img_ch=3, output_ch=1

        ngf = 64
        filters = [ngf, ngf * 2, ngf * 4, ngf * 8, ngf * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.Conv1 = SPADEEordBlock(input_nc, filters[0], self.opt)
        self.Conv2 = SPADEEordBlock(filters[0], filters[1], self.opt)
        self.Conv3 = SPADEEordBlock(filters[1], filters[2], self.opt)


        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = SPADEEordBlock(filters[2], filters[1], self.opt)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = SPADEEordBlock(filters[1], filters[0], self.opt)

        self.Conv = nn.Conv2d(filters[0], output_nc, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Sigmoid()


    def forward(self, x, extra = None, getatt = False):

        # if self.opt.mix_input_gen:
        #     x = torch.cat([x,extra], dim = 1)

        e1 = self.Conv1(x, extra)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2, extra)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3, extra)

        #print(x5.shape)
        d3 = self.Up3(e3)
        #print(d5.shape)
        if getatt:
            atts= []

            x2, att3 = self.Att3(g=d3, x=e2, fakeatt = self.fakeatt, getatt = getatt)
            atts.append(att3)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3, extra)

            d2 = self.Up2(d3)
            x1, att2 = self.Att2(g=d2, x=e1, fakeatt = self.fakeatt, getatt = getatt)
            atts.append(att2)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2, extra)

            out = self.Conv(d2)

            out = self.active(out)


            return out, atts
        else:

            x2 = self.Att3(g=d3, x=e2, fakeatt = self.fakeatt)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_conv3(d3, extra)

            d2 = self.Up2(d3)
            x1 = self.Att2(g=d2, x=e1, fakeatt = self.fakeatt)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv2(d2, extra)

            out = self.Conv(d2)

            out = self.active(out)

            return out

class SesameInterventor(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=2, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=4, help='number of residual blocks in the global Interventor network')
        parser.add_argument('--spade_n_blocks', type=int, default=5, help='number of residual blocks in the global Interventor network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        output_nc = 3
        label_nc = opt.label_nc

        input_nc = label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
        if opt.mix_input_gen:
            input_nc += 4

        norm_layer = get_nonspade_norm_layer(opt, 'instance')
        activation = nn.ReLU(False)


        # initial block 
        self.init_block = nn.Sequential(*[nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation])
        
        # Downsampling blocks
        self.downlayers = nn.ModuleList()
        mult = 1
        for i in range(opt.resnet_n_downsample):
            self.downlayers.append(nn.Sequential(*[norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]))
            mult *= 2

        # Semantic core blocks
        self.resnet_core = nn.ModuleList()
        if opt.wide: 
            self.resnet_core += [ResnetBlock(opt.ngf * mult,
                                dim2=opt.ngf * mult * 2,
                                norm_layer=norm_layer,
                                activation=activation,
                                kernel_size=opt.resnet_kernel_size)]
            mult *= 2
        else:
            self.resnet_core += [ResnetBlock(opt.ngf * mult,
                                norm_layer=norm_layer,
                                activation=activation,
                                kernel_size=opt.resnet_kernel_size)]


        for i in range(opt.resnet_n_blocks - 1):
            self.resnet_core += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size,
                                  dilation=2)]


        self.spade_core = nn.ModuleList()
        for i in range(opt.spade_n_blocks - 1):
            self.spade_core += [SPADEResnetBlock(opt.ngf * mult, opt.ngf * mult, opt, dilation=2)]

        if opt.wide:
            self.spade_core += [SPADEResnetBlock(opt.ngf * mult * (2 if not self.opt.no_skip_connections else 1), opt.ngf * mult//2, opt)]
            mult//=2
        else:
            self.spade_core += [SPADEResnetBlock(opt.ngf * mult * (2 if not self.opt.no_skip_connections else 1), opt.ngf * mult, opt)]

        # Upsampling blocks
        self.uplayers = nn.ModuleList()
        for i in range(opt.resnet_n_downsample):
            self.uplayers.append(SPADEResnetBlock(mult * opt.ngf * (3 if not self.opt.no_skip_connections else 2)//2, opt.ngf * mult//2, opt))
            mult //= 2

        final_nc = opt.ngf


        self.conv_img = nn.Conv2d((input_nc + final_nc) if not self.opt.no_skip_connections else final_nc , output_nc, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)


    def forward(self, input, extra=None):
        if self.opt.mix_input_gen:
            input = torch.cat([input,extra], dim = 1)
        
        dec_i = self.opt.resnet_n_downsample 
        skip_connections = []
        # InitBlock
        x = self.init_block(input)
        skip_connections.append(x)
        # /InitBlock

        # Downsize
        for downlayer in self.downlayers:
            x = downlayer(x)
            skip_connections.append(x)
        # /Downsize

        # SemanticCore 
        for res_layer in self.resnet_core:
            x = res_layer(x)


        for spade_layer in self.spade_core[:-1]:
            x = spade_layer(x, extra)
            
        if not self.opt.no_skip_connections:
            x = torch.cat([x, skip_connections[dec_i]],dim=1)
            dec_i -= 1
    
        x = self.spade_core[-1](x, extra)
        # /SemanticCore 

        # Upsize
        for uplayer in self.uplayers:
            x = self.up(x)
            if not self.opt.no_skip_connections:
                x = torch.cat([x, skip_connections[dec_i]],dim=1)
                dec_i -= 1
            x = uplayer(x, extra)
        # /Upsize


        # OutBlock
        if not self.opt.no_skip_connections:
            x = torch.cat([x, input],dim=1)
        x = self.conv_img(F.leaky_relu(x, 2e-1))

        x = F.tanh(x)
        # /OutBlock

        return x

class unetInterventor(BaseNetwork):
    """Create a Unet-based Interventor"""

    def __init__(self, opt):
        """Construct a Unet Interventor
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(unetInterventor, self).__init__()
        self.opt = opt
        label_nc = opt.label_nc

        input_nc = label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
        if opt.mix_input_gen:
            input_nc += 4
        output_nc = 3
        num_downs = 6
        ngf=64
        norm_layer=nn.BatchNorm2d
        use_dropout=False
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input, extra = None):
        """Standard forward"""
        if self.opt.mix_input_gen:
            input = torch.cat([input,extra], dim = 1)
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)