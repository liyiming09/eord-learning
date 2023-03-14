"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import multiprocessing
multiprocessing.set_start_method('spawn', True)

import torch.nn.functional as f
import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from models.learneord_model import LearnEordModel
from models.onlye_model import OnlyEModel
from util.visualizer import Visualizer
from util import html
import torch.distributed as dist
import numpy as np
from skimage.measure import compare_ssim
from skimage.color import rgb2gray

def main():
    dist.init_process_group(backend='nccl')
    opt = TestOptions().parse()
    
    # dataloader = data.create_dataloader(opt)
    # load the dataset
    dataset = data.create_dataset(opt)

     # create trainer for our model
    # trainer = Pix2PixTrainer(opt)
    model = OnlyEModel(opt)
    # load the dataloader
    dataloader = data.create_dataloader(opt, dataset)

    model.eval()

    visualizer = Visualizer(opt)

    # create a webpage that summarizes the all results
    prefix = 'removal' if opt.removal else 'addtion'
    web_dir = os.path.join(opt.results_dir, opt.name,
                        '%s_%s_%s' % (prefix, opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir,
                        'Experiment = %s, Phase = %s, Epoch = %s' %
                        (opt.name, opt.phase, opt.which_epoch))

    # test
    ssim = []
    acc = []
    for i, data_i in enumerate(dataloader):
        if i * opt.batchSize >= opt.how_many:
            break

        masked_image, semantics_seq = model(data_i, mode='interventor_test')

        if masked_image.shape[1] != 3:
            masked_image = masked_image[:,:3]
        
        semantics = semantics_seq[0]
        
                   #note1!!!!!!!!!!!!1
        img_path = data_i['path']
        if opt.bbox:
            img_path = data_i['image_path']
        try: 
            ran =  semantics.shape[0]
        except:
            ran =  semantics[0].shape[0]

        for b in range(ran):
            # print('process image... %s' % img_path[b])
            label = data_i['label'][b]
            if opt.segmentation_mask and not opt.phase == "test":
                label = semantics[b].unsqueeze(0).max(dim=1)[1]
    
            mask = semantics[b][-1]
            mask_true_neg = semantics_seq[5][b:b+1]
            mask_pos, mask_neg = semantics_seq[6][b:b+1], semantics_seq[7][b:b+1]    #note1!!!!!!!!!!!!1

            visuals = OrderedDict([('input_label', label),
                                ('mask', data_i['mask_in'][b]),
                                # ('synthesized_image', generated[b]),
                                ('real_label', data_i['label'][b]),
                                ('intervention_pos', mask_pos), ('intervention_neg', mask_neg), ('eord_neg', mask_true_neg),                  #note2!!!!!!!!!!!!
                                # ('image_pos', pos), ('image_neg', neg),
                                ('real_image', data_i['image'][b]),
                                # ('attention_map', atts),
                                ('masked_image', masked_image[b])])

            if not opt.no_instance:
                instance = data_i['instance'][b]
                if opt.segmentation_mask: 
                    instance = semantics[b,35].unsqueeze(0)

                visuals['instance'] = instance

            
            if opt.tf_log:
                visualizer.display_current_results(visuals, 200, b, True)
            else:
                visualizer.save_images(webpage, visuals, img_path[b:b + 1],i)

            # Compute SSIM on edited areas
            # pred_img = generated[0].detach().cpu().numpy().transpose(1,2,0)
            # gt_img = data_i['image'].float()[0].numpy().transpose(1,2,0)
            # pred_img = rgb2gray(pred_img)
            # # print(pred_img.max(),pred_img.mean(),pred_img.min())
            # gt_img = rgb2gray(gt_img)
            # ssim_pic = compare_ssim(gt_img,pred_img, multichannel=False, full=True)[1]

            # mask = data_i['mask_in'][0]
            # ssim.append(np.ma.masked_where(1 - mask.cpu().numpy().squeeze(), ssim_pic).mean())

    webpage.save()
    # print(np.mean(ssim))

if __name__ == '__main__':
    main()
