#!/usr/bin/python3
# coding:utf-8

"""
    Author: Yifei Chen
    Email: chenyifei14@huawei.com
    License: MIT
    This is a python file for combining and visualizing results of all the models we have in the task: frontalization (RaR, PsP, Ours)
"""


import os
from PIL import Image
import sys
from torchvision import transforms
def generate_dirname(root):
    dir_list = os.listdir(root)
    with open('dir_vox1_name.txt','a+') as f:
        for name in dir_list:
            f.writelines(name+'\n')


# FEI
psp = '/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel/experiment/fei_all_psp_50000/inference_results/'
rar = '/mnt/nas6/users/xiesong/code/3D/Rotate-and-Render-master/FEI_results/rs_model/example/orig_rename'
# our = '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/frontalization/3dmmrt_610e_1015/FEI_Face/'
our = '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/frontalization/3dmmrt_symmetric_e220_1025/FEI_Face/'
source = '/mnt/nas6/users/xiesong/data/3D/FEI_Face/test_data'
combine = '/mnt/nas7/users/chenyifei/data/paper_visualization/frontalization/FEI/'

# FF
# psp = '/mnt/nas7/users/chenyifei/code/humanface/pixel2style2pixel/experiment/FF_all_psp_50000/inference_results/'
# rar = '/mnt/nas6/users/xiesong/code/3D/Rotate-and-Render-master/FaceForensics_results/rs_model/example/orig'
# our = '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/frontalization/3dmmrt_symmetric_e220_1025/FaceForensics/'
# source = '/mnt/nas7/users/chenyifei/data/FaceForensics_test_frontal/'
# combine = '/mnt/nas7/users/chenyifei/data/paper_visualization/frontalization/FaceForenics/'

img_transform = transforms.Compose([transforms.Scale(256),
                                     transforms.CenterCrop(256)])
if __name__ == '__main__':
    fnames = os.listdir(source)
    os.makedirs(combine,exist_ok=True)
    global_i = 0
    for fname in fnames:
        if os.path.isfile(os.path.join(source,fname)):
            if global_i % 50 == 0:
                print('processing', fname)
            list_img=[
                os.path.join(source,fname),
                os.path.join(psp, fname),
                os.path.join(rar, fname),
                os.path.join(our, fname),
            ]

            new_im = Image.new('RGB', (256*4,256))

            for ind,elem in enumerate(list_img):
                im = Image.open(elem)
                im = img_transform(im)
                new_im.paste(im, (ind*256, 0))
            new_im.save(os.path.join(combine, fname))
            global_i += 1

    print('Done with %d combination' %global_i)