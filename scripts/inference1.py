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

    # --- metric counting ------------------------------------
    sim_threshold = 0.2
    angle_threshold = 15
    total_sim_fit = 0
    total_angle_fit = 0
    total_full_fit = 0
    avg_id_loss = 0
    # -------------------------------------------------------
    transformations = transforms.Compose([transforms.Scale(256),
                                          transforms.CenterCrop(256),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    id_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                       ])
    #---------------- moco calculater
    device = 'cuda:0'
    moco_loss_calculator = moco_loss.MocoLoss().to(device).eval()
    id_loss_calculator = id_loss.IDLoss().to(device).eval()
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
    # --- DHP model -----------------------------------
    print('loading deep head pose model')
    hope_net = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    dhp_snapshot_path = '/mnt/nas7/users/chenyifei/code/humanface/deep-head-pose/pretrained_models/hopenet_robust_alpha1.pkl'
    dhp_saved_state_dict = torch.load(dhp_snapshot_path)
    hope_net.load_state_dict(dhp_saved_state_dict)
    hope_net.cuda()
    hope_net.eval()


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
        sim_fit = 0
        angle_fit = 0
        full_fit = 0
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
            #- Identity Loss -----------


            # ----try resize to 256
            result_batch_resize = F.interpolate(result_batch, size=256)
            # ---------------------------

            loss_moco, moco_sim_improvement, moco_logs = moco_loss_calculator(result_batch_resize, gt_cuda, input_cuda)  # result_batch: inference   y:gt    x:input
            loss_id, id_sim_improvement, id_logs = id_loss_calculator(result_batch_resize, gt_cuda, input_cuda)
            print('Batch {}:'.format(batch_idx))
            print('    Moco: loss {:.4f}    sim_imporve {:.4f}'.format(loss_moco.item(),moco_sim_improvement))
            print('    Identity: loss {:.4f}    sim_imporve {:.4f}'.format(loss_id.item(), id_sim_improvement))
            # if loss_moco.item()<sim_threshold:
            #     sim_fit += 1
            avg_id_loss += loss_id.item()
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(opts.test_batch_size):
            result = tensor2im(result_batch[i])
            # im_path = test_dataset.source_paths[global_i]
            im_path = dataset.paths[global_i]
            if opts.couple_outputs or global_i % 5 == 0:
                # input_im = log_input_image(input_batch[i], opts)
                input_im = log_input_image(input_cuda[i], opts)
                resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
                if opts.resize_factors is not None:
                    # for super resolution, save the original, down-sampled, and output
                    source = Image.open(im_path)
                    res = np.concatenate([np.array(source.resize(resize_amount)),
                                          np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
                                          np.array(result.resize(resize_amount))], axis=1)
                else:
                    # otherwise, save the original and output
                    res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                          np.array(result.resize(resize_amount))], axis=1)
                Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            Image.fromarray(np.array(result)).save(im_save_path)



            # temporarily block eval part cause we deprecate it via python file sim_angle_eval.py
            #---- Angle estimator-------
            # img_dhp = Image.open(im_save_path)
            # img_dhp = img_dhp.convert('RGB')
            # img_dhp = transformations(img_dhp)
            # img_dhp = img_dhp.unsqueeze(0)
            # img_dhp = img_dhp.cuda()
            # yaw, pitch, roll = hope_net(img_dhp)
            # _, yaw_bpred = torch.max(yaw.data, 1)
            # _, pitch_bpred = torch.max(pitch.data, 1)
            # _, roll_bpred = torch.max(roll.data, 1)
            # yaw_predicted = util_dhp.softmax_temperature(yaw.data, 1)
            # pitch_predicted = util_dhp.softmax_temperature(pitch.data, 1)
            # roll_predicted = util_dhp.softmax_temperature(roll.data, 1)
            # idx_tensor = [idx for idx in range(66)]
            # idx_tensor = torch.FloatTensor(idx_tensor).cuda()
            # yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
            # pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
            # roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99
            # if abs(yaw_predicted.item())<angle_threshold and abs(pitch_predicted.item())<angle_threshold and abs(roll_predicted.item())<angle_threshold:
            #     angle_fit += 1
            # if sim_fit != 0 and angle_fit != 0:
            #     full_fit += 1
            #
            # total_sim_fit += sim_fit
            # total_angle_fit += angle_fit
            # total_full_fit += full_fit
            # print('    Yaw:{:.4f}    Pitch:{:.4f}    Roll:{:.4f}\n'.format(yaw_predicted.item(),pitch_predicted.item(),roll_predicted.item()))
            # ------------------------------------------------------------



            global_i += 1
    avg_id_loss /= global_i

    # temporarily block eval part cause we deprecate it via python file sim_angle_eval.py
    #----- calculate percentile
    # print('ID/SIM: {:.2f}%    Angle: {:.2f}%    Both: {:.2f}%'.format(total_sim_fit*100/global_i,
    #                                                                   total_angle_fit*100/global_i,
    #                                                                   total_full_fit*100/global_i))



    print('Avg id_loss: {:.2f}'.format(avg_id_loss))
    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)



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
