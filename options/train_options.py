"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=2000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=40, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=60, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')

        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--lambda_style', type=float, default=250.0, help='weight for style loss')
        parser.add_argument('--lambda_effect', type=float, default=10.0, help='weight for divco loss')
        parser.add_argument('--lambda_divco', type=float, default=0, help='weight for divco loss')
        parser.add_argument('--lambda_ms', type=float, default=0, help='weight for modeseek loss')
        parser.add_argument('--lambda_monce', type=float, default=10, help='weight for monce loss')
        parser.add_argument('--lambda_recons', type=float, default=100, help='weight for monce loss')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--use_style_loss', action='store_true', help='if specified, do use style loss')
        parser.add_argument('--recons_loss', action='store_true', help='if specified, do use recons_loss loss')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--netD', type=str, default='sesamemultiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--netF', type=str, default='Patch', help='(n_layers|multiscale|image)')
        parser.add_argument('--cost_type', type=str, default='hard', help='(hard or easy)')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
        parser.add_argument('--lambda_kld', type=float, default=0.05)


        parser.add_argument('--tau', type=float, default=1, help='temprature')
        parser.add_argument('--num_negative', type=int, default=2, help='# of NEGATIVE SAMPLES during the aug')
        parser.add_argument('--featnorm', action='store_true', help='whether featnorm')
        parser.add_argument('--ot_weight', type=float, default=256.0, help='weight for vgg loss')
       
        
        self.isTrain = True
        return parser
