import os
from PIL import Image
def run(file_root,dest_dir_root,sample_rate):
    os.makedirs(dest_dir_root,exist_ok=True)
    file_txt = open(file_root, 'r')
    image_paths = file_txt.readlines()
    count = 0

    floder_set = {}
    for i,image_path in enumerate(image_paths):
        image_path = image_path.split('\n')[0]
        path_parts = image_path.split('/')
        sub_dir = path_parts[-2]
        if sub_dir not in floder_set:
            floder_set[sub_dir] = 0
        else:
            if i % sample_rate != 0 or floder_set[sub_dir] >= 10:
                continue
        print('processing',image_path)
        img = Image.open(image_path)
        save_name = path_parts[-2]+'_'+path_parts[-1]
        save_path = os.path.join(dest_dir_root,save_name)
        img.save(save_path)
        floder_set[sub_dir] = floder_set[sub_dir]+1
        count += 1
    print('Successfully read and save %d images'%count)


if __name__ == '__main__':
    sample_rate = 50
    file_root = "/mnt/nas7/users/zhousai/FaceForensics/faceforensics_large_angle_pose.txt"
    dest_dir_root = '/mnt/nas7/users/chenyifei/data/FaceForensics_test_frontal/'
    run(file_root,dest_dir_root,sample_rate)