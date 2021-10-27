import os
from PIL import Image
def run(file_root,dest_dir_root):
    os.makedirs(dest_dir_root,exist_ok=True)
    file_txt = open(file_root, 'r')
    image_paths = file_txt.readlines()
    count = 0
    for image_path in image_paths:
        image_path = image_path.split('\n')[0]
        print('processing',image_path)
        img = Image.open(image_path)
        path_parts = image_path.split('/')
        save_name = path_parts[-2]+'_'+path_parts[-1]
        save_path = os.path.join(dest_dir_root,save_name)
        img.save(save_path)
        count += 1
    print('Successfully read and save %d images'%count)




if __name__ == '__main__':
    file_root = '/mnt/nas7/users/zhousai/FaceForensics/facefornsics_large_angle_pose.txt'
    dest_dir_root = '/mnt/nas7/users/chenyifei/data/FaceForensics_test_frontal/'
    run(file_root,dest_dir_root)

