"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import multiprocessing
multiprocessing.set_start_method('spawn', True)

import torch.nn.functional as f
import os
from collections import OrderedDict

import data, ntpath
from options.test_options import TestOptions
from models.pix2pixa_model import Pix2PixAModel

from util.visualizer import Visualizer
from util import html
import util
import numpy as np
from skimage.measure import compare_ssim
from skimage.color import rgb2gray


def convert_visuals_to_numpy(visuals, b):
    for key, t in visuals.items():
        tile = True
        if 'input_label' == key:
            #t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)
            t = util.util.tensor2label(t, 255, tile=tile)
        elif 'real_label' == key:
            t = np.transpose(t.cpu().float().numpy(), (1, 2, 0)).astype(np.uint8)
        elif 'attention_map' == key:
            maps = []
            for att in t:
                tmp = att[b].cpu().squeeze().float().numpy()
                tmp = (tmp*255).astype(np.uint8).astype(np.uint8)
                maps.append(tmp)
            t = maps
        else:
            t = util.util.tensor2im(t, tile=tile)
        visuals[key] = t
    return visuals

# save image to the disk
def save_attimages( webpage, visuals, image_path, b):        
    visuals = convert_visuals_to_numpy(visuals, b)        
    
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0] + str(b)

    for label, image_numpy in visuals.items():
        if label == 'attention_map':
            for i in range(4):
                new_name = name + '_' + str(i)
                image_name = os.path.join(label, '%s.png' % (new_name))
                save_path = os.path.join(image_dir, image_name)
                util.util.save_image(image_numpy[i], save_path, create_dir=True)
        else:
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(image_dir, image_name)
            util.util.save_image(image_numpy, save_path, create_dir=True)



def main():
    opt = TestOptions().parse()
    # model = Pix2PixAModel(opt)# load the dataset
    dataset = data.create_dataset(opt)

     # create trainer for our model
    # trainer = Pix2PixTrainer(opt)
    model = Pix2PixAModel(opt)
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

        generated, masked_image, semantics, atts = model(data_i, mode='inference')

        if masked_image.shape[1] != 3:
            masked_image = masked_image[:,:3]
        


        img_path = data_i['path']
        if opt.bbox:
            img_path = data_i['image_path']
        try: 
            ran =  generated.shape[0]
        except:
            ran =  generated[0].shape[0]

        # print(data_i['label'].shape)
        # print(len(atts))
        for b in range(ran):
            # print('process image... %s' % img_path[b])
            label = data_i['label'][b]

            #将inputlabel 转换为仅有编辑区域
            if opt.segmentation_mask and not opt.phase == "test":
                label = semantics[b].unsqueeze(0).max(dim=1)[1]
    
            mask = semantics[b][-1]
            visuals = OrderedDict([('input_label', label),
                                ('synthesized_image', generated[b]),
                                ('real_label', data_i['label'][b]),
                                ('real_image', data_i['image'][b]),
                                ('attention_map', atts),
                                ('masked_image', masked_image[b])])

            if not opt.no_instance:
                instance = data_i['instance'][b]
                if opt.segmentation_mask: 
                    instance = semantics[b,35].unsqueeze(0)

                visuals['instance'] = instance

            
            if opt.tf_log:
                visualizer.display_current_results(visuals, 200, b, True)
            else:
                # visualizer.save_images(webpage, visuals, img_path[b:b + 1],i)
                # save_images(webpage, visuals, img_path[b:b + 1],b)
                save_attimages(webpage, visuals, img_path[b:b + 1],b)

            # Compute SSIM on edited areas
            # pred_img = generated[0].detach().cpu().numpy().transpose(1,2,0)
            # gt_img = data_i['image'].float()[0].numpy().transpose(1,2,0)
            # pred_img = rgb2gray(pred_img)
            # gt_img = rgb2gray(gt_img)
            # ssim_pic = compare_ssim(gt_img,pred_img, multichannel=False, full=True)[1]

            # mask = data_i['mask_in'][0]
            # ssim.append(np.ma.masked_where(1 - mask.cpu().numpy().squeeze(), ssim_pic).mean())

    webpage.save()
    # print(np.mean(ssim))

if __name__ == '__main__':
    main()
