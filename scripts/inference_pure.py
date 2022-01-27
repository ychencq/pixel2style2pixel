'''
running example
python scripts/inference1.py \
--exp_dir=/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel/experiment/psp_test_fei_180000/ \
--checkpoint_path=experiment/frontal_train_ffhq_batch8/checkpoints/iteration_180000.pt \
--data_path=/mnt/nas6/users/xiesong/data/3D/FEI_Face/test_data \
--test_batch_size=1 \
--test_workers=4
'''

import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

sys.path.append("/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel")
import torch.nn.functional as F

from criteria import id_loss, w_norm,moco_loss
from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp
from datasets.images_dataset import ImagesDataset

import models.hopenet as hopenet
import  utils.util_dhp as util_dhp

path = "/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel/"

def run():


    # -------------------------------------------------------
    transformations = transforms.Compose([transforms.Scale(256),
                                          transforms.CenterCrop(256),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # ---------------
    test_opts = TestOptions().parse()

    if test_opts.resize_factors is not None:
        assert len(
            test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
                                        'downsampling_{}'.format(test_opts.resize_factors))
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
                                        'downsampling_{}'.format(test_opts.resize_factors))
    else:
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(path+test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 256
    opts = Namespace(**opts)
    # ---- psp model -----------------------------------
    net = pSp(opts)
    net.eval()
    net.cuda()
    # ----- Dataset -------------------------------------
    print('Loading dataset for {}'.format(opts.dataset_type))
    print('data path: {}'.format((opts.data_path)))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()

    dataset = InferenceDataset(root=opts.data_path,
                               # transform=transforms_dict['transform_inference'],
                               transform = transformations,
                               opts=opts)

    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)
    # test_dataset = ImagesDataset(source_root=opts.data_path,
    #                              target_root=opts.data_path,
    #                              source_transform=transforms_dict['transform_source'],
    #                              target_transform=transforms_dict['transform_test'],
    #                              opts=opts)
    #
    # test_dataloader = DataLoader(test_dataset,
    #                         batch_size=opts.test_batch_size,
    #                         shuffle=False,
    #                         num_workers=int(opts.test_workers),
    #                         drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    # ---- inference ---------------------------------------
    global_i = 0
    global_time = []
    for batch_idx,input_batch in enumerate(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch
            gt_cuda = input_batch
            input_cuda = input_cuda.cuda().float()
            gt_cuda = gt_cuda.cuda().float()
            # input_cuda = input_cuda.unsqueeze(0)
            # gt_cuda = gt_cuda.unsqueeze(0)
            tic = time.time()
            # - inference --------------
            result_batch = run_on_batch(input_cuda, net, opts)
            # ----try resize to 256
            result_batch_resize = F.interpolate(result_batch, size=256)
            # ---------------------------
            print('Batch {} processed'.format(batch_idx))
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(opts.test_batch_size):
            result = tensor2im(result_batch[i])
            # im_path = test_dataset.source_paths[global_i]
            im_path = dataset.paths[global_i]
            parts = os.path.basename(im_path).split('.npy_')
            save_sub_dir = parts[0]
            im_save_path = os.path.join(out_path_results, parts[0])
            os.makedirs(im_save_path,exist_ok=True)
            Image.fromarray(np.array(result)).save(os.path.join(im_save_path,parts[1]))
            global_i += 1

def run_on_batch(inputs, net, opts):
    if opts.latent_mask is None:
        result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject,
                      alpha=opts.mix_alpha,
                      resize=opts.resize_outputs)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch


if __name__ == '__main__':
    run()
