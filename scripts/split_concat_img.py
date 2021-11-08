from PIL import Image
import os

def split(img_list,dest_dir):
    os.makedirs(dest_dir,exist_ok=True)
    part_list = ['original','-45','-30','-15','0','15','30','45']
    for img in img_list:
        print('processing', img.split('/')[-1])
        im_name = img.split('/')[-1].split('.png')[0]
        im = Image.open(img)
        width, height = im.size
        width = width//8
        split_list = []
        for i,part in enumerate(part_list):
            im_part = im.crop((i*width,0,(i+1)*width,height))
            im_part.save(os.path.join(dest_dir,im_name+'_'+part+'.png'))


if __name__ == '__main__':
    #rotate x
    # dest_dir = '/mnt/nas7/users/chenyifei/data/3dmmrt_rotate/generated/rotate_x/'
    # img_list = ['/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_x/3dmmrt_610e_1015/vox/'
    #             'id10283#uA1E_38TuTw#002329#002480.npy_22.png',
    #             '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_x/3dmmrt_610e_1015/vox/'
    #             'id10285#m-uILToQ9ss#004267#004374.npy_27.png',
    #             '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_x/3dmmrt_610e_1015/vox/'
    #             'id10283#u3s9xdUlmmk#000798#001159.npy_43.png',
    #             '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_x/3dmmrt_610e_1015/vox/'
    #             'id10283#r9-0pljhZqs#014722#014831.npy_66.png',
    #             '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_x/3dmmrt_610e_1015/vox/'
    #             'id10283#vaK4t1-WD4M#021373#021976.npy_42.png']
    # split(img_list,dest_dir)

    # rotate y
    # dest_dir = '/mnt/nas7/users/chenyifei/data/3dmmrt_rotate/generated/rotate_y/'
    # img_list = ['/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_y/3dmmrt_610e_1015/vox/'
    #             'id10291#iPZGPcrYIZs#002071#002715.npy_26.png',
    #             '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_y/3dmmrt_610e_1015/vox/'
    #             'id10291#TMCTm7GxiDE#002595#002901.npy_87.png',
    #             '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_y/3dmmrt_610e_1015/vox/'
    #             'id10292#ENIHEvg_VLM#017575#017702.npy_50.png',
    #             '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_y/3dmmrt_610e_1015/vox/'
    #             'id10291#4aLg_keiGHw#001217#001424.npy_84.png',
    #             '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_y/3dmmrt_610e_1015/vox/'
    #             'id10285#m-uILToQ9ss#004267#004374.npy_27.png',
    #             '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_y/3dmmrt_610e_1015/vox/'
    #             'id10283#vaK4t1-WD4M#021249#021369.npy_94.png',
    #             '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_y/3dmmrt_610e_1015/vox/'
    #             'id10282#neQO6_CUY4w#000765#001023.npy_93.png',
    #             '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_y/3dmmrt_610e_1015/vox/'
    #             'id10292#57KbI2UvGss#010365#010549.npy_19.png',
    #             '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_y/3dmmrt_610e_1015/vox/'
    #             'id10291#4aLg_keiGHw#001497#001678.npy_80.png',
    #             '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_y/3dmmrt_610e_1015/vox/'
    #             'id10291#uiBjIKX_0l8#000067#000277.npy_76.png',
    #             '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_y/3dmmrt_610e_1015/vox/'
    #             'id10290#gTbcgoYXWdU#001513#001814.npy_15.png'
    #             ]

    # rotate z
    dest_dir = '/mnt/nas7/users/chenyifei/data/3dmmrt_rotate/generated/rotate_z/'
    img_list = ['/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_z/3dmmrt_610e_1015/vox/'
                'id10291#uiBjIKX_0l8#001197#001806.npy_7.png',
                '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_z/3dmmrt_610e_1015/vox/'
                'id10283#uA1E_38TuTw#003095#003227.npy_6.png',
                '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_z/3dmmrt_610e_1015/vox/'
                'id10283#vv4mvANXHcs#006391#007149.npy_906.png',
                '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_z/3dmmrt_610e_1015/vox/'
                'id10289#sf4uMnkYFG8#006920#007944.npy_241.png',
                '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_z/3dmmrt_610e_1015/vox/'
                'id10281#ni6gO5jDLJE#004685#004875.npy_151.png',
                '/mnt/nas7/datasets/public/CV/virtual_human/3dmmrt/experiments/generated/rotate_iterative/rotate_z/3dmmrt_610e_1015/vox/'
                'id10285#FUqAFZmZJ80#008142#008336.npy_231.png'
                ]
    split(img_list,dest_dir)