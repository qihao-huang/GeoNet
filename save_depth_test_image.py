import os
import shutil

file_name = "/userhome/34/h3567721/projects/Depth/GeoNet/data/kitti/test_files_eigen.txt"
dataset_dir = "/userhome/34/h3567721/dataset/kitti/raw_data/"
save_path = "/userhome/34/h3567721/projects/Depth/GeoNet/log/depth_test_set/"

with open(file_name, 'r') as f:
    test_files = f.readlines()
    test_files = [dataset_dir + t[:-1] for t in test_files]

for i, img in enumerate(test_files):
    shutil.copy2(img, save_path+str(i)+".png")