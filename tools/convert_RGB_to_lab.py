import os
import cv2
import glob
from shutil import copyfile

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

base_dir = "/userhome/34/h3567721/dataset/kitti"
kitti_raw_eigen_dir = os.path.join(base_dir, "kitti_depth_test_eigen")
kitti_raw_eigen_lab_dir = os.path.join(base_dir, "kitti_depth_test_eigen_lab")
make_dir(kitti_raw_eigen_lab_dir)
all_dirs = [o for o in os.listdir(kitti_raw_eigen_dir) if os.path.isdir(os.path.join(kitti_raw_eigen_dir,o))]
# os.path.join(base_dir, o) 

for img_dir in all_dirs:
    print("processing ", img_dir)
    img_path_list = glob.glob(os.path.join(kitti_raw_eigen_dir, img_dir) + "/*.jpg")
    cam_txt_path_list = glob.glob(os.path.join(kitti_raw_eigen_dir, img_dir) + "/*.txt")

    make_dir(os.path.join(kitti_raw_eigen_lab_dir, img_dir))

    for i in range(len(img_path_list)):
        src_img_path = img_path_list[i]
        rgb_img = cv2.imread(src_img_path)
        lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB)
        dst_img_path = src_img_path.replace("kitti_depth_test_eigen", "kitti_depth_test_eigen_lab")
        cv2.imwrite(dst_img_path, lab_img)

        src_cam_path = cam_txt_path_list[i]
        dst_cam_path = src_cam_path.replace("kitti_depth_test_eigen", "kitti_depth_test_eigen_lab")
        copyfile(src_cam_path, dst_cam_path)