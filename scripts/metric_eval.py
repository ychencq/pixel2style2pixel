import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
import torchvision

sys.path.append(".")
sys.path.append("..")

from criteria import id_loss, w_norm,moco_loss
from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp
from datasets.images_dataset import ImagesDataset
import models.hopenet as hopenet
import  utils.util_dhp as util_dhp


def run():
    #--------Moco and Identity
    device = 'cuda:0'
    moco_loss_calculator = moco_loss.MocoLoss().to(device).eval()
    id_loss_calculator = id_loss.IDLoss().to(device).eval()
    # -------DHP
    print('loading deep head pose model')
    hope_net = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    dhp_snapshot_path = '/mnt/nas7/users/chenyifei/code/humanface/deep-head-pose/pretrained_models/hopenet_robust_alpha1.pkl'
    dhp_saved_state_dict = torch.load(dhp_snapshot_path)
    hope_net.load_state_dict(dhp_saved_state_dict)
    # ------dataloader
    test_opts = TestOptions().parse()
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    print('Loading dataset for {}'.format(opts.dataset_type))
    print('data path: {}'.format((opts.data_path)))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()

    eval_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
                                 target_root=dataset_args['test_target_root'],
                                 source_transform=transforms_dict['transform_source'],
                                 target_transform=transforms_dict['transform_test'],
                                 opts=opts)

    test_dataloader = DataLoader(eval_dataset,
                                 batch_size=opts.test_batch_size,
                                 shuffle=False,
                                 num_workers=int(opts.test_workers),
                                 drop_last=True)



    loss_moco, moco_sim_improvement, moco_logs = moco_loss_calculator(result_batch, gt_cuda,input_cuda)  # result_batch: inference   y:gt    x:input
    loss_id, id_sim_improvement, id_logs = id_loss_calculator(result_batch, gt_cuda, input_cuda)
    print('Batch {}:'.format(batch_idx))
    print('    Moco: loss--{:.4f}    sim--{:.4f}    logs--{}'.format(loss_moco.item(), moco_sim_improvement, moco_logs))
    print('    Identity: loss--{:.4f}    sim-{:.4f}    logs-{}\n'.format(loss_id.item(), id_sim_improvement, id_logs))
    # print('    Moco: loss--{:.4f}    sim--{:.4f}'.format(loss_moco.item(),moco_sim_improvement,moco_logs))
    # print('    Identity: loss--{:.4f}    sim-{:.4f}\n'.format(loss_id.item(), id_sim_improvement, id_logs))