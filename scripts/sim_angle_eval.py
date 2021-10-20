"""
    File name: sim_angle_eval.py
    Author: Yifei Chen
    email: chenyifei14@huawei.com
    Date created: 10/20/2021
    Date last modified: 10/20/2021
"""
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

sys.path.append("/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel")
from utils import data_utils
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

# --- Dataloader ------------------------------------------------------
class EvalDataset(Dataset):
	def __init__(self, inference_root, gt_root, transform=None):
		self.inference_paths = sorted(data_utils.make_dataset(inference_root))
		self.gt_paths = sorted(data_utils.make_dataset(gt_root))
		self.transform = transform

	def __len__(self):
		return len(self.inference_paths)

	def __getitem__(self, index):
		infer_path = self.inference_paths[index]
		infer_img = Image.open(infer_path)
		infer_img = infer_img.convert('RGB') #if self.opts.label_nc == 0 else infer_img.convert('L')

		gt_path = self.gt_paths[index]
		gt_img = Image.open(gt_path)
		gt_img = gt_img.convert('RGB') #if self.opts.label_nc == 0 else gt_img.convert('L')


		if self.transform: # transfer to tensor as well as normalize
			gt_img = self.transform(gt_img)
			infer_img = self.transform(infer_img)
		infer_name = infer_path.split('/')[-1]
		gt_name = gt_path.split('/')[-1]
		if infer_name != gt_name:
			raise ValueError('Ooops! inference & ground truth dont match.')

		return infer_img, gt_img, infer_name
# ---------------------------------------------------------------------

# --- Run Function ------------------------------------------------------
def run(inference_root,gt_root,save_root):
	# metric counting
	sim_threshold = 0.3
	angle_threshold = 15
	total_sim_fit = 0
	total_angle_fit = 0
	total_full_fit = 0
	id_transform = transforms.Compose([transforms.Resize((256, 256)),
									   transforms.ToTensor(),
									   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
	dhp_transform = transforms.Compose([transforms.Scale(224),
										  transforms.CenterCrop(224),
										  transforms.ToTensor(),
										  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	# Moco & identity estimator model
	device = 'cuda:0'
	moco_loss_calculator = moco_loss.MocoLoss().to(device).eval()
	print('loading identity loss model')
	id_loss_calculator = id_loss.IDLoss().to(device).eval()

	# DHP: Angle estimator model
	print('loading deep head pose model')
	hope_net = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
	dhp_snapshot_path = '/mnt/nas7/users/chenyifei/code/humanface/deep-head-pose/pretrained_models/hopenet_robust_alpha1.pkl'
	dhp_saved_state_dict = torch.load(dhp_snapshot_path)
	hope_net.load_state_dict(dhp_saved_state_dict)
	hope_net.cuda()
	hope_net.eval()

	# save root/dir statement
	out_path_coupled = os.path.join(save_root, 'gt_inference_coupled') # to check if infer match gt
	os.makedirs(out_path_coupled, exist_ok=True)

	# Dataset and Dataloader
	print('loading infer dataset: {}'.format(inference_root))
	print('loading gt dataset: {}'.format(gt_root))

	dataset = EvalDataset(inference_root = inference_root,
						  gt_root = gt_root,
						  transform=id_transform)
	dataloader = DataLoader(dataset,
							batch_size=1,
							shuffle=False,
							num_workers=4,
							drop_last=True)
	total_samples = len(dataset)

	#Eval
	global_i = 0
	for batch_idx, input_batch in enumerate(dataloader):
		# metric init in every iteration
		sim_fit = 0
		angle_fit = 0
		full_fit = 0
		if global_i >= total_samples:
			break
		with torch.no_grad():
			# SIM/ID calculate
			infer_cuda,gt_cuda,name = input_batch
			infer_cuda = infer_cuda.cuda().float() #try change later
			gt_cuda = gt_cuda.cuda().float()
			name = name[0]
			# print(infer_cuda.shape)
			# print(gt_cuda.shape)
			loss_moco, moco_sim_improvement, moco_logs = moco_loss_calculator(infer_cuda, gt_cuda,gt_cuda)  # result_batch:inference y:gt x:input
			loss_id, id_sim_improvement, id_logs = id_loss_calculator(infer_cuda, gt_cuda, gt_cuda)
			print('Sample {}:'.format(name))
			print('    Moco: loss {:.4f}    sim_imporve {:.4f}'.format(loss_moco.item(), moco_sim_improvement))
			print('    Identity: loss {:.4f}    sim_imporve {:.4f}'.format(loss_id.item(), id_sim_improvement))
			# metric counting
			if loss_moco.item() < sim_threshold:
				sim_fit += 1

		# img save for checking matching
		infer_im = tensor2im(infer_cuda[0])
		gt_im = tensor2im(gt_cuda[0])
		if global_i % 5 == 0:
			resize_amount = (256, 256)
			res = np.concatenate([np.array(infer_im.resize(resize_amount)),
								  np.array(gt_im.resize(resize_amount))], axis=1)
			Image.fromarray(res).save(os.path.join(out_path_coupled, name))

		# Angle estimator
		im_path = dataset.inference_paths[global_i]
		img_dhp = Image.open(im_path)
		img_dhp = img_dhp.convert('RGB')
		img_dhp = dhp_transform(img_dhp) # already to tensor
		img_dhp = img_dhp.unsqueeze(0).cuda()
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
		print('    Yaw:{:.4f}    Pitch:{:.4f}    Roll:{:.4f}\n'.format(yaw_predicted.item(),
																	   pitch_predicted.item(),
																	   roll_predicted.item()))
		# ------------------------------------------------------------
		# metric counting
		if abs(yaw_predicted.item()) < angle_threshold and abs(pitch_predicted.item()) < angle_threshold and abs(
				roll_predicted.item()) < angle_threshold:
			angle_fit += 1
		if sim_fit != 0 and angle_fit != 0:
			full_fit += 1

		total_sim_fit += sim_fit
		total_angle_fit += angle_fit
		total_full_fit += full_fit
		global_i += 1
	print('Total samples evaluated:{}'.format(global_i))
	print('ID/SIM: {:.2f}%    Angle: {:.2f}%    Both: {:.2f}%'.format(total_sim_fit * 100 / global_i,
																	  total_angle_fit * 100 / global_i,
																	  total_full_fit * 100 / global_i))


if __name__ == '__main__':
	gt_root = '/mnt/nas7/users/chenyifei/data/FEI_testmini/images/'
	inference_root = '/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel/experiment/fei_testmini/inference_results/'
	save_root = '/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel/experiment/fei_testmini/check_match/'
	# gt_root = '/mnt/nas6/users/xiesong/data/3D/FEI_Face/test_data/'
	# gt_root = '/mnt/nas6/users/xiesong/data/3D/FEI_Face/test_gt/'
	# inference_root = '/mnt/nas6/users/xiesong/code/3D/Rotate-and-Render-master/FEI_results/rs_model/example/orig_rename'
	# save_root = '/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel/experiment/fei_xiesong/check_match/'
	# gt_root = '/mnt/nas6/users/xiesong/data/3D/mmgen/real_imgs/'
	# inference_root = '/mnt/nas6/users/xiesong/data/3D/mmgen/fake_imgs/'
	# save_root = '/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel/experiment/shiyong/check_match/'
	run(inference_root,gt_root,save_root)


