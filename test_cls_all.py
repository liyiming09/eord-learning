"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import multiprocessing
multiprocessing.set_start_method('spawn', True)

import torch.nn.functional as f
import os, ntpath
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from models.learneord_model import LearnEordModel
from models.cls_model import ClsModel
from util.visualizer import Visualizer
from util import util
import torch.distributed as dist
import numpy as np
from skimage.measure import compare_ssim
from skimage.color import rgb2gray

def main():
    # dist.init_process_group(backend='nccl')
    opt = TestOptions().parse()
    
    # dataloader = data.create_dataloader(opt)
    # load the dataset
    dataset = data.create_dataset(opt)

     # create trainer for our model
    # trainer = Pix2PixTrainer(opt)
    model = ClsModel(opt)
    # load the dataloader
    dataloader = data.create_dataloader_noddp(opt, dataset)

    model.eval()
    # create a webpage that summarizes the all results

    for i, datas in enumerate(dataloader):

        for j, data_i in enumerate(datas):
            if data_i['cls'][0][0] == None or data_i['cls'][0][0] == 34: continue
            maps, masked_image, semantics_seq = model(data_i, mode='gen_C_set')
            cls = data_i['cls'][0][0].numpy()
            bbox = data_i['bbox']
            if masked_image.shape[1] != 3:
                masked_image = masked_image[:,:3]
            
            semantics = semantics_seq[1]
            
                    #note1!!!!!!!!!!!!1
            img_path = data_i['path']
            if opt.bbox:
                img_path = data_i['image_path']
            b = 0
            label = data_i['label'][b]
            if opt.segmentation_mask and not opt.phase == "test":
                label = semantics[b]

            visuals = OrderedDict([('input_label', label),
                                # ('mask', data_i['mask_in'][b]),
                                ('intpret_map', maps[0]), ('intpret_map_rgb', maps[1])
                                ])

            for key, t in visuals.items():
                if key == 'input_label':
                    t = (np.transpose(t.detach().cpu().float().numpy(), (1, 2, 0))* 255).astype(np.uint8)
                elif 'intpret' in key:
                    t = (t*255).astype(np.uint8).squeeze()
                visuals[key] = t
            
            short_path = ntpath.basename(img_path[0])
            name = os.path.splitext(short_path)[0] + '_' + str(j)


            res_dir = os.path.join(opt.results_dir, opt.name, opt.phase)
            for key, t in visuals.items():
                image_name = os.path.join(key, str(cls), '%s.png' % (name))
                save_path = os.path.join(res_dir, image_name)
                util.save_image(t, save_path, create_dir=True)


            # Compute SSIM on edited areas
            # pred_img = generated[0].detach().cpu().numpy().transpose(1,2,0)
            # gt_img = data_i['image'].float()[0].numpy().transpose(1,2,0)
            # pred_img = rgb2gray(pred_img)
            # # print(pred_img.max(),pred_img.mean(),pred_img.min())
            # gt_img = rgb2gray(gt_img)
            # ssim_pic = compare_ssim(gt_img,pred_img, multichannel=False, full=True)[1]

            # mask = data_i['mask_in'][0]
            # ssim.append(np.ma.masked_where(1 - mask.cpu().numpy().squeeze(), ssim_pic).mean())

    # print(np.mean(ssim))

if __name__ == '__main__':
    main()
