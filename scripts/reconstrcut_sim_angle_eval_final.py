"""
    File name: sim_angle_eval.py
    Author: Yifei Chen
    email: chenyifei14@huawei.com
    Date created: 10/26/2021
    Date last modified: 10/26/2021
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

from models.mtcnn.mtcnn import MTCNN
from models.encoders.model_irse import IR_101
from configs.paths_config import model_paths
CIRCULAR_FACE_PATH = model_paths['circular_face']


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
			print(infer_name)
			print(gt_name)
			raise ValueError('Ooops! inference & ground truth dont match.')

		return infer_img, gt_img, infer_name
# ---------------------------------------------------------------------

# --- Run Function ------------------------------------------------------
def run(inference_root,gt_root,save_root):
	# metric counting
	avg_moco_score = 0
	avg_arc_id_score = 0
	avg_curr_id_score = 0

	# id_transform = transforms.Compose([transforms.Resize((256, 256)),
	# 								   transforms.ToTensor(),
	# 								   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
	data_transform = transforms.Compose([transforms.Scale(256),
									   transforms.CenterCrop(256),
									   transforms.ToTensor(),
									   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
	facenet_id_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
	# Moco & identity estimator model
	device = 'cuda:0'
	moco_loss_calculator = moco_loss.MocoLoss().to(device).eval()
	print('loading arcface identity loss model')
	id_loss_calculator = id_loss.IDLoss().to(device).eval()

	# Id loss model
	print('loading curricular identity loss model')
	facenet = IR_101(input_size=112)
	facenet.load_state_dict(torch.load(CIRCULAR_FACE_PATH))
	facenet.cuda()
	facenet.eval()
	mtcnn = MTCNN()
	# print('\t{} is starting to extract on {} images'.format(pid, len(file_paths)))
	# tot_count = len(file_paths)
	# count = 0

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
	print('infer dataset path: {}'.format(inference_root))
	print('gt dataset path: {}'.format(gt_root))

	dataset = EvalDataset(inference_root = inference_root,
						  gt_root = gt_root,
						  transform=data_transform)
	# dataloader = DataLoader(dataset,
	# 						batch_size=1,
	# 						shuffle=False,
	# 						num_workers=4,
	# 						drop_last=True)
	total_samples = len(dataset)

	#Eval
	global_i = 0
	for i in range(total_samples):
		if global_i >= total_samples:
			break
		infer_name = dataset.inference_paths[global_i].split('/')[-1]
		gt_name = dataset.gt_paths[global_i].split('/')[-1]
		if infer_name != gt_name:
			raise ValueError('Ooops! inference {} & ground truth {} dont match.'.format(infer_name,gt_name))
		
		with torch.no_grad():
			# ---------------------------------------------------------------------------------
			# temporarly block dataloader for MOCO and ArcFace
			res_path = dataset.inference_paths[global_i]
			input_im = Image.open(res_path)
			input_im, _ = mtcnn.align(input_im)  # align to (112,112)
			input_im = facenet_id_transform(input_im).unsqueeze(0).cuda()
			
			gt_path = dataset.gt_paths[global_i]
			result_im = Image.open(gt_path)
			result_im, _ = mtcnn.align(result_im)  # align to (112,112)
			result_im = facenet_id_transform(result_im).unsqueeze(0).cuda()

			loss_moco, moco_sim_improvement, moco_logs = moco_loss_calculator(input_im, result_im, result_im)
			arc_loss_id, arc_id_sim_improvement, id_logs = id_loss_calculator(input_im, result_im, result_im)

			input_id = facenet(input_im)[0]
			result_id = facenet(result_im)[0]
			curr_id_score = float(input_id.dot(result_id))
			# -----------------------------------------------------------------------------------
			print('Sample {}:'.format(infer_name))
			print('    Moco: score {:.4f} '.format(1-loss_moco.item() ))
			print('    ArcId: Score {:.4f} '.format(1-arc_loss_id.item()))
			print('    CurrId: Score {:.4f} '.format(curr_id_score))

			# metric cal
			avg_moco_score += 1-loss_moco.item()
			avg_arc_id_score += 1-arc_loss_id.item()
			avg_curr_id_score += curr_id_score

		# img save for checking matching
		gt_path = dataset.gt_paths[global_i]
		gt_img = Image.open(gt_path)
		gt_img = gt_img.convert('RGB') #if self.opts.label_nc == 0 else gt_img.convert('L')
		infer_path = dataset.inference_paths[global_i]
		infer_img = Image.open(infer_path)
		infer_img = infer_img.convert('RGB')

		gt_cuda = data_transform(gt_img).cuda().float()
		infer_cuda = data_transform(infer_img).cuda().float()

		infer_im = tensor2im(infer_cuda)
		gt_im = tensor2im(gt_cuda)
		if global_i % 50 == 0:
			resize_amount = (256, 256)
			res = np.concatenate([np.array(infer_im.resize(resize_amount)),
								  np.array(gt_im.resize(resize_amount))], axis=1)
			Image.fromarray(res).save(os.path.join(out_path_coupled, infer_name))

		# ------------------------------------------------------------
		global_i += 1  #assume batch size = 1

	avg_moco_score /= global_i
	avg_arc_id_score /= global_i
	avg_curr_id_score /= global_i
	print('Total samples evaluated:{}'.format(total_samples))
	print('Avg moco_socre: {:.2f}'.format(avg_moco_score))
	print('Avg Arc id_score: {:.2f}'.format(avg_arc_id_score))
	print('Avg Curricular id_score: {:.2f}'.format(avg_curr_id_score))


if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "5"
	# check psp size 不对应的情况下 计算正确性
	# gt_root = '/mnt/nas7/users/chenyifei/data/ffhq_256_mini/'
	# inference_root = '/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel/experiment/ffhq256_testmini/inference_results/'
	# save_root = '/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel/experiment/ffhq256_testmini/check_match/'
	#

	# check 指标计算正确性
	# gt_root = '/mnt/nas6/users/xiesong/data/3D/FEI_Face/test_gt/'
	# inference_root = '/mnt/nas6/users/xiesong/data/3D/FEI_Face/test_data/'
	# save_root = '/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel/experiment/fei_all_ValidateAlgo/check_match/'


	# xiesong RaR
	# gt_root = '/mnt/nas6/users/xiesong/data/3D/FEI_Face/test_gt/'
	# inference_root = '/mnt/nas6/users/xiesong/code/3D/Rotate-and-Render-master/FEI_results/rs_model/example/orig_rename/'
	# save_root = '/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel/experiment/fei_all_RaR/check_match/'


	# yifei PsP
	# gt_root = '/mnt/nas6/users/xiesong/data/3D/FEI_Face/test_gt/'
	# inference_root = '/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel/experiment/fei_all_psp_200000/inference_results/'
	# save_root = '/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel/experiment/fei_all_psp_200000/check_match/'


	# shiyong 3dmmrt
	gt_root = '/mnt/nas6/users/xiesong/data/3D/FEI_Face/test_gt/'
	inference_root = '/mnt/nas7/users/chenyifei/code/humanface/mmgeneration/frontalization_experiment/fei_test_all/inference_results/'
	save_root = '/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel/experiment/fei_all_3dmmrt/check_match/'

	# yifei x2face -- pose2face ?
	run(inference_root,gt_root,save_root)


