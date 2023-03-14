import cv2,os,math, sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
from skimage.measure.simple_metrics import compare_psnr

import os
import pathlib

import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from collections import namedtuple

from skimage.color import rgb2gray
from pytorch_fid.inception import InceptionV3
sys.path.append("..") 
from segmentation_models.segment import DRNSeg
from segmentation_models import segmodels
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

cls_map = np.load("map.npy", allow_pickle = True).item()

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

class AdeSegPathDataset(torch.utils.data.Dataset):
    def __init__(self, res_files, label_files):
        self.files = res_files
        self.label_files = label_files
        self.normalize = TF.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        # self.img_transform = TF.Compose([
        #     # torchvision.transforms.Resize(512),
        #     TF.ToTensor(),
        #     TF.Resize(512),
        #     self.normalize
        # ])
        # self.label_transforms = TF.Compose([
        #     # torchvision.transforms.Resize(512),
        #     TF.ToTensor(),
        #     TF.Resize(512)
        # ])
        self.imgSizes = [300, 375, 450, 525, 600]
        self.imgMaxSize = 1000
        self.padding_constant = 32
    
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def __len__(self):
        return len(self.files)

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def label_transforms(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long()
        return segm

    def __getitem__(self, i):
        # print(self.files[i],self.label_files[i] )
        img_path = self.files[i]
        img = Image.open(img_path).convert('RGB')
        ori_width, ori_height = img.size
        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            # img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        img = [x.contiguous() for x in img_resized_list]
        # img = self.img_transform(img)
        label_path = self.label_files[i]
        segm = Image.open(label_path).convert('L')

        
        # segm = torch.from_numpy(np.array(segm)).long() - 1
        # array = np.array(label)
        # out_array = np.empty(array.shape, dtype=array.dtype)
        # for l in labels:
        #     out_array[array == l.id] = l.trainId
        # label = Image.fromarray(out_array)
        if self.label_transforms is not None:
            segm = self.label_transforms(segm)
            # segm = self.label_transforms(segm) * 255 - 1
        name = img_path.name
        return img, segm, name

class CitySegPathDataset(torch.utils.data.Dataset):
    def __init__(self, res_files, label_files):
        self.files = res_files
        self.label_files = label_files
        info = {"std": [0.1829540508368939, 0.18656561047509476, 0.18447508988480435], "mean": [0.29010095242892997, 0.32808144844279574, 0.28696394422942517]}
        normalize = TF.Normalize(mean=info['mean'], std=info['std'])
        self.transforms = TF.Compose([
            # torchvision.transforms.Resize(512),
            TF.ToTensor(),
            normalize
        ])
        self.label_transforms = TF.Compose([
            # torchvision.transforms.Resize(512),
            TF.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        # print(self.files[i],self.label_files[i] )
        img_path = self.files[i]
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        label_path = self.label_files[i]
        label = Image.open(label_path).convert('L')

        array = np.array(label)
        out_array = np.empty(array.shape, dtype=array.dtype)
        for l in labels:
            out_array[array == l.id] = l.trainId
        label = Image.fromarray(out_array)
        if self.label_transforms is not None:
            label = self.label_transforms(label)[0] * 255
        name = img_path.name
        return img, label, name

def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=0):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers)

    return m, s

def compute_index( pred, gt, n_cl):
    assert (pred.shape == gt.shape)  # gt的形状和pred的形状必须相等
    k = (gt >= 0) & (gt < n_cl)  # 生成一个H×W的mask，里面类别属于0~n_cl的为True，否则为False
    labeled = np.sum(k)  # 统计mask中元素为True的数量，即标签类别是0~n_cl的元素个数
    correct = np.sum((pred[k] == gt[k]))  # 统计预测正确的数量
    # 返回混淆矩阵、统计元素的数量和预测正确元素的数量。
    hist = np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),minlength=n_cl ** 2).reshape(n_cl, n_cl)


    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  # 交并比 = 对角线元素 / 所有元素和 -对角线元素
    mean_IU = np.nanmean(iu)  # numpy.nanmean()函数可用于计算忽略NaN值的数组平均值,经过这一步就算得了MIOU
    mean_IU_no_back = np.nanmean(iu[1:])  # 除去背景的MIOU（一般情况下0类别代表背景
    mean_pixel_acc = correct / labeled  # 平均像素准确率 = 预测正确像素个数 / 总像素个数
    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc

def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

def compute_miou_acc(res_roots, gt_mask_root,label_roots, segmodel, batch_size, device, num_classes = 19, num_workers=0):
    res_path = pathlib.Path(res_roots)
    res_files = sorted([file for ext in IMAGE_EXTENSIONS for file in res_path.glob('*.{}'.format(ext))])
    mask_path = pathlib.Path(gt_mask_root)
    mask_files = sorted([file for ext in IMAGE_EXTENSIONS for file in mask_path.glob('*.{}'.format(ext))])
    if '-ade' in res_files[0].parts[1]:
        dataset = AdeSegPathDataset(res_files, mask_files)
    else:
        dataset = CitySegPathDataset(res_files, mask_files)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    mious = []
    accs = []
    miou_nobgs = []
    newacc = 0
    nums = 0
    segmodel.eval()
    data = tqdm(dataloader)
    for image, label, name in data:
        with torch.no_grad():
            if segmodel._get_name() == 'ademodel':
                
                label = label.to(device)
                segSize = (label.shape[1], label.shape[2])
                final_mask = torch.zeros(1, num_classes, segSize[0], segSize[1]).cuda()
                for img in image:
                    inputimg = img.to(device)
                    scores_tmp = segmodel(inputimg, segSize = segSize)
                    final_mask = final_mask + scores_tmp / len(dataset.imgSizes)

            else:
                image = image.to(device)
                label = label.to(device)
                final_mask = segmodel(image)[0]
        _, pred = torch.max(final_mask, 1)
        pred = pred[0].cpu().data.numpy()

        label = label[0].cpu().numpy()

        if segmodel._get_name() == 'ademodel':
            

            sesame_cls = np.unique(pred+1)
            array = np.array(pred+1)
            out_array = np.empty(array.shape, dtype=array.dtype)
            for cur_cls in sesame_cls:
                if cur_cls  in cls_map.keys():
                    out_array[array == cur_cls] = cls_map[cur_cls]
                else:
                    out_array[array == cur_cls] = 100

            pred = out_array
        #仅计算roi内的miou和acc
        if miou_roi:
            label_name = os.path.join(label_roots,name[0])
            smask = cv2.imread(label_name)
            mask = (smask[:,:,0] != 0 ) | ( smask[:,:,1] != 0 ) | ( smask[:,:,2] != 128)
            mask = mask.astype(int)

            pred_roi = np.ma.masked_where(1 - mask, pred)
            label_roi = np.ma.masked_where(1 - mask, label)
        else:
            pred_roi = pred
            label_roi = label

        # hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
        if segmodel._get_name() == 'ademodel':
            acc, pix = accuracy(pred_roi, label_roi)
            newacc += acc * pix
            nums += pix
            iou, miou_nobg, miou , acc = compute_index(pred_roi.flatten(), label_roi.flatten(), num_classes+1)
        else:
            iou, miou, miou_nobg, acc = compute_index(pred_roi.flatten(), label_roi.flatten(), num_classes)
        
        # print("miou: ", miou)
        # print("acc: ", acc)
        data.set_description("miou: " + str(miou)+ ",acc: "+ str( acc))
        mious.append(miou)
        accs.append(acc)
        miou_nobgs.append(miou_nobg)
        torch.cuda.empty_cache()
    
    if segmodel._get_name() == 'ademodel': print("new acc:", newacc/nums)
    return np.nanmean(mious), np.nanmean(accs)
flag_fid = True
flag_ssim = True
flag_miou = True
city_classes = 19
ade_classes = 150
miou_roi = False

# def ca

def cal_index(root, base, total, models):
    index= {}

    baseroot = os.path.join(root ,base)
    res_dirs = os.listdir(baseroot)
    res_dirs.sort()
    sub_root = 'images'# final_inpainting   MyInpainting  DBII inpainting-RES
    # res_roots = os.path.join(baseroot,model_name,'results/')
    # res_roots = 
    dims = 2048
    batch_size = 1
    device = 'cuda:7'
    if flag_fid:
        
        
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

        fidmodel = models['fid']
    # print(res_dirs)
    if flag_miou:
        if '-ade' in base:
            # print(base)
            segmodel = models['ade']
            classes = ade_classes
        else:
            segmodel = models['city']
            classes = city_classes

            
    
    mssim = {}
    for sdir in res_dirs:
        if base in total and  sdir  in total[base]:
            index[sdir] = total[base][sdir]
        else:
            index[sdir] = {}
            # phase = sdir.split('_')[0]

            
        gt_roots = os.path.join(baseroot,sdir,sub_root,'real_image')
        res_roots = os.path.join(baseroot,sdir,sub_root,'synthesized_image')
        label_roots = os.path.join(baseroot,sdir,sub_root,'input_label')
        smask_roots = os.path.join(baseroot,sdir,sub_root,'mask')
        gt_mask_root = os.path.join(baseroot,sdir,sub_root,'real_label')
        #计算fid：
        if flag_fid:
            if 'fid' not in index[sdir]:
                m1, s1 = compute_statistics_of_path(res_roots, fidmodel, batch_size,
                                            dims, device, num_workers=0)
                m2, s2 = compute_statistics_of_path(gt_roots, fidmodel, batch_size,
                                                    dims, device, num_workers=0)
                fid_value = calculate_frechet_distance(m1, s1, m2, s2)
                print(base, sdir, fid_value)
                index[sdir]['fid'] = fid_value
        #计算mIoU ACC：
        if flag_miou:
            if ('miou' not in index[sdir]) or index[sdir]['miou'] <= 0.2:
                miou, acc = compute_miou_acc(res_roots, gt_mask_root, label_roots, segmodel, batch_size, device, num_classes = classes, num_workers=0 )
                index[sdir]['miou'] = miou
                index[sdir]['acc'] = acc
                print("------------", base, sdir)
                print("final miou: ", np.mean(miou))
                print("final macc: ", np.mean(acc))
        #cal ssim
        if flag_ssim:
            if 'ssim' not in index[sdir] or index[sdir]['ssim'] == np.nan or index[sdir]['ssim'] > 0.7:
                mssim[sdir] = []
                images = os.listdir(res_roots)
                images.sort()
                length = len(images)
                #计算ssim psnr
                for i in range(length):
                    fake_name = os.path.join(res_roots, images[i])
                    gt_name = os.path.join(gt_roots,images[i])
                    label_name = os.path.join(label_roots,images[i])
                    smask_name = os.path.join(smask_roots,images[i])
                    gt = cv2.imread(gt_name)
                    fake= cv2.imread(fake_name)
                    label = cv2.imread(label_name)

                    if 'hong' in base:
                        smask = cv2.imread(smask_name,0)
                        smask = smask!=127
                        mask = smask.astype(int)
                    else:
                        mask = (label[:,:,0] != 0 ) | ( label[:,:,1] != 0 ) | ( label[:,:,2] != 128)
                        mask = mask.astype(int)
                #     psnr1 = psnr(gt,gt)
                    fake = (fake/255 - 0.5)*2
                    gt = (gt/255 - 0.5)*2
                    fake = rgb2gray(fake)
                    gt = rgb2gray(gt)
                    
                    # psnr = compare_psnr(gt,fake,255)
                    if mask.sum() != 0:
                        wssim = compare_ssim(gt,fake, multichannel=False, full=True)[1]
                        # print(np.ma.masked_where(1 - mask, wssim))
                        ssim = np.ma.masked_where(1 - mask, wssim)
                        # print(ssim.max())
                    else:
                        ssim = compare_ssim(gt,fake, multichannel=False, full=True)[1]
                    ssim = ssim.mean()

                    mssim[sdir].append(ssim)

            #         print(images[i*6], psnr,ssim)

                print(base, sdir,np.mean(mssim[sdir]) )#np.mean(mpsnr[sdir]),
                index[sdir]['ssim'] = np.mean(mssim[sdir])
    return index
def Ademodel():
    torch.cuda.set_device(7)

    # Network Builders
    net_encoder = segmodels.build_encoder(
        arch="resnet101",
        fc_dim=2048,
        weights='./../segmentation_models/encoder_epoch_50.pth')
    net_decoder = segmodels.build_decoder(
        arch="upernet",
        fc_dim=2048,
        num_class=150,
        weights='./../segmentation_models/decoder_epoch_50.pth',
        use_softmax=True)

    return segmodels.ademodel(net_encoder, net_decoder)
if __name__ == '__main__':

    if os.path.exists('index.npy'):
        total = np.load('index.npy', allow_pickle=True).item()
        for key, item in total.items():
            print(key)
            for skey, sitem in item.items():
                print( '\t \t',skey, '\t \t', sitem)
    else:
        total = {}
    # total.pop('refine-G-7-effect-masknce=0.1-pos=cos-style500-fakeatt')
    root = './results'
    root_dirs = os.listdir(root)
    root_dirs.sort()

    device = 'cuda:7'
    models = {}
    if flag_fid:
        dims = 2048
        
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

        fidmodel = InceptionV3([block_idx]).to(device)
        models['fid'] = fidmodel
    # print(res_dirs)
    if flag_miou:

        arch = 'drn_d_105'
        
        citymodel = segmodels.DRNSeg(arch, city_classes, pretrained_model=None,
                        pretrained=False).to(device)
        citymodel.load_state_dict(torch.load('./../segmentation_models/drn-d-105_ms_cityscapes.pth'))
        models['city'] = citymodel
        # ademodel.load_state_dict(torch.load('./../segmentation_models/drn-d-105_ms_cityscapes.pth'))
        models['ade'] = Ademodel().cuda()

    for item in root_dirs:
        # if "-ade" not in item : continue
        if "-ED" in item : continue
        total[item] = cal_index(root, item, total, models)

        np.save('index.npy',total)
