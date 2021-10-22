from torchvision.transforms import ToTensor, Compose, Scale
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
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
import numpy as np
import torch.nn as nn
import os
import torch
from PIL import Image
from torch.autograd import Variable
import pickle
import sys
sys.path.append("/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel")


import models.hopenet as hopenet
import  utils.util_dhp as util_dhp


def load_from_npy(path):
    real_numpy = np.load(path)
    real_tensor = torch.tensor(real_numpy)
    source_img = Variable(real_tensor).cuda()  # B C H W
    return source_img

def run(real_path,filename,dest_dir):
    os.makedirs(dest_dir,exist_ok=True)

    # DHP: Angle estimator model
    # load model
    print('loading deep head pose model')
    hope_net = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    dhp_snapshot_path = '/mnt/nas7/users/chenyifei/code/humanface/deep-head-pose/pretrained_models/hopenet_robust_alpha1.pkl'
    dhp_saved_state_dict = torch.load(dhp_snapshot_path)
    hope_net.load_state_dict(dhp_saved_state_dict)
    hope_net.cuda()
    hope_net.eval()
    # transform prepared for hopenet
    dhp_transform = transforms.Compose([transforms.Scale(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # read file
    f = open(filename, 'rb')
    x = pickle.load(f)
    keys = [key for key in x.keys()]
    global_num_frame = 0
    global_num_video = 0
    for key in keys:
        names = key.split('.')
        file_numpy = names[0] + '.npy'
        realimg_path = real_path + file_numpy
        source_np = np.load(realimg_path) # B C H W
        # source_img = load_from_npy(realimg_path)
        source_num = np.size(source_np,0)
        infer_list = []  # [[yaw, pitch, roll]]
        for start in range(0, source_num, 1):
            end = min(start + 1, source_num)
            batch_size = end - start
            source_batch = source_np[start:end, :, :, :]
            source_batch = np.squeeze(source_batch, axis=0)
            # transform
            source_batch = (source_batch * 255).astype(np.uint8)
            source_batch = source_batch.transpose(1, 2, 0)
            img_dhp = Image.fromarray(source_batch)
            img_dhp = img_dhp.convert('RGB')
            img_dhp = dhp_transform(img_dhp)  # already to tensor
            img_dhp = img_dhp.unsqueeze(0).cuda()
            # inference
            with torch.no_grad():
                yaw, pitch, roll = hope_net(img_dhp)
            _, yaw_bpred = torch.max(yaw.data, 1)
            _, pitch_bpred = torch.max(pitch.data, 1)
            _, roll_bpred = torch.max(roll.data, 1)
            yaw_predicted = util_dhp.softmax_temperature(yaw.data, 1)
            pitch_predicted = util_dhp.softmax_temperature(pitch.data, 1)
            roll_predicted = util_dhp.softmax_temperature(roll.data, 1)
            idx_tensor = [idx for idx in range(66)]
            idx_tensor = torch.FloatTensor(idx_tensor).cuda()
            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99
            if start%50 == 0:
                print('{}-{:d}    Yaw:{:.4f}    Pitch:{:.4f}    Roll:{:.4f}'.format(names[0],
                                                                                      start,
                                                                                      yaw_predicted.item(),
                                                                                      pitch_predicted.item(),
                                                                                      roll_predicted.item()))
            infer = [yaw_predicted.item(),pitch_predicted.item(),roll_predicted.item()]
            infer_list.append(infer)
            global_num_frame += 1
        global_num_video += 1
        print('\n')
        save_path = os.path.join(dest_dir, file_numpy)
        np.save(save_path, infer_list)
    print('Totally processed {} videos'.format(global_num_video))
    print('Totally processed {} frames'.format(global_num_frame))


if __name__ == '__main__':
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    # CUDA_VISIBLE_DEVICES = 0, 1
    real_path = '/mnt/nas3/users/lishiyong/logs/mmgen/first_order_vox1_with_3dmm_rt_eye_101301/eval_samples/real_imgs/'
    filename = '/mnt/nas3/users/chendapeng/results_analysis/results/video_L1_e1_total.pkl'
    dest_dir = '/mnt/nas7/users/chenyifei/data/x2face_trial/pose_estimate'
    run(real_path,filename,dest_dir)

