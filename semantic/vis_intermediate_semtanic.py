import os
import numpy as np

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def vis_depth(base_path, vis_img, vis=False):
    vis_depth_dir = os.path.join(base_path, "vis_depth")
    make_dir(vis_depth_dir)
    depth_img = vis_img
    # --------------------------------------------- 
    print("depth_img: ", depth_img.shape)         # (12, 128, 416, 1)
    for i in range(depth_img.shape[0]):
        sig_depth = depth_img[i, :, :, 0] 
        # print(sig_depth.shape) # shape (128, 416)
        vis_save_path = os.path.join(vis_depth_dir, os.path.splitext(file_name)[0]+"-"+str(i)+".png")

        fig = plt.gcf()
        plt.imshow(1.0/sig_depth, cmap="plasma")
        plt.show()
        fig.savefig(vis_save_path)

def vis_pose(base_path, vis_img, vis=False):
    vis_pose_dir = os.path.join(base_path, "vis_pose")
    make_dir(vis_pose_dir)
    pose_file = vis_img
    # --------------------------------------------- 
    print("pose_file: ", pose_file.shape)       # (4, 2, 6)
    for i in range(pose_file.shape[0]):
        sig_pose_1 = pose_file[i,0,:]
        sig_pose_2 = pose_file[i,1,:]
        vis_save_path_1 = os.path.join(vis_pose_dir, os.path.splitext(file_name)[0]+"-"+str(i)+"-1.txt")
        vis_save_path_2 = os.path.join(vis_pose_dir, os.path.splitext(file_name)[0]+"-"+str(i)+"-2.txt")
        np.savetxt(vis_save_path_1, sig_pose_1, delimiter=',')
        np.savetxt(vis_save_path_2, sig_pose_2, delimiter=',')


def vis_delta(base_path, vis_img, vis=False):
    vis_delta_dir = os.path.join(base_path, "vis_delta")
    make_dir(vis_delta_dir)
    delta_img = vis_img

     # --------------------------------------------- 
    # for deltax_xyz, assumption: 
    # i=0: 0:3 -> fwd, 6:9 -> bwd
    # i=1: 3:6 -> fwd, 9:12 -> bwd

    print("delta_img: ", delta_img.shape)         # (4, 128, 416, 12)
    for i in range(delta_img.shape[0]):
        sig_delta_concat = delta_img[i, :, :, :] 
        # print(sig_delta_concat.shape) # shape (128, 416, 12)

        sig_delta_1 = sig_delta_concat[:,:,0:3]
        # norm_1 = mpl.colors.Normalize(vmin = np.min(sig_delta_1), vmax = np.max(sig_delta_1), clip = False)
        vis_save_path_1 = os.path.join(vis_delta_dir, os.path.splitext(file_name)[0]+"-"+str(i)+"-0-3"+".png")
        fig = plt.gcf()
        plt.imshow(sig_delta_1/100)
        plt.show()
        fig.savefig(vis_save_path_1)

        sig_delta_2 = sig_delta_concat[:,:,3:6]
        # norm_2 = mpl.colors.Normalize(vmin = np.min(sig_delta_2), vmax = np.max(sig_delta_2), clip = False)
        vis_save_path_2 = os.path.join(vis_delta_dir, os.path.splitext(file_name)[0]+"-"+str(i)+"-3-6"+".png")
        fig = plt.gcf()
        plt.imshow(sig_delta_2/100)
        plt.show()
        fig.savefig(vis_save_path_2)

        sig_delta_3 = sig_delta_concat[:,:,6:9]
        # norm_3 = mpl.colors.Normalize(vmin = np.min(sig_delta_3), vmax = np.max(sig_delta_3), clip = False)
        vis_save_path_3 = os.path.join(vis_delta_dir, os.path.splitext(file_name)[0]+"-"+str(i)+"-6-9"+".png")
        fig = plt.gcf()
        plt.imshow(sig_delta_3/100)
        plt.show()
        fig.savefig(vis_save_path_3)

        sig_delta_4 = sig_delta_concat[:,:,9:12]
        # norm_4 = mpl.colors.Normalize(vmin = np.min(sig_delta_4), vmax = np.max(sig_delta_4), clip = False)
        vis_save_path_4 = os.path.join(vis_delta_dir, os.path.splitext(file_name)[0]+"-"+str(i)+"-9-12"+".png")
        fig = plt.gcf()
        plt.imshow(sig_delta_4/100)
        plt.show()
        fig.savefig(vis_save_path_4)

def vis_tgt(base_path, vis_img, vis=False):
    vis_tgt_dir = os.path.join(base_path, "vis_tgt")
    make_dir(vis_tgt_dir)
    tgt_image = vis_img

    print("tgt_image: ", tgt_image.shape)         # (4, 128, 416, 3)
    for i in range(tgt_image.shape[0]):
        sig_tgt = tgt_image[i, :, :, :] 
        # print(sig_tgt.shape) # shape (128, 416, 3)
        vis_save_path = os.path.join(vis_tgt_dir, os.path.splitext(file_name)[0]+"-"+str(i)+".png")

        fig = plt.gcf()
        plt.imshow(sig_tgt)
        plt.show()
        fig.savefig(vis_save_path)


def vis_tgt_sem(base_path, vis_img, vis=False):
    vis_tgt_sem_dir = os.path.join(base_path, "vis_tgt_sem")
    make_dir(vis_tgt_sem_dir)
    tgt_sem_image = vis_img

    print("tgt_sem_image: ", tgt_sem_image.shape)         # (4, 128, 416, 3)
    for i in range(tgt_sem_image.shape[0]):
        sig_tgt = tgt_sem_image[i, :, :, :] 
        # print(sig_tgt.shape) # shape (128, 416, 3)
        vis_save_path = os.path.join(vis_tgt_sem_dir, os.path.splitext(file_name)[0]+"-"+str(i)+".png")

        fig = plt.gcf()
        plt.imshow(sig_tgt)
        plt.show()
        fig.savefig(vis_save_path)


def vis_src(base_path, vis_img, vis=False):    
    vis_src_dir = os.path.join(base_path, "vis_src")
    make_dir(vis_src_dir)
    src_image = vis_img

    print("src_image: ", src_image.shape)         # (4, 128, 416, 6)

    for i in range(src_image.shape[0]):
        sig_src_1 = src_image[i, :, :, :3] 
        # print(sig_src_1.shape) # shape (128, 416, 3)
        sig_src_2 = src_image[i, :, :, 3:6] 
        # print(sig_src_2.shape) # shape (128, 416, 3)
        vis_save_path_1 = os.path.join(vis_src_dir, os.path.splitext(file_name)[0]+"-"+str(i)+"-1.png")
        vis_save_path_2 = os.path.join(vis_src_dir, os.path.splitext(file_name)[0]+"-"+str(i)+"-2.png")

        fig = plt.gcf()
        plt.imshow(sig_src_1)
        plt.show()
        fig.savefig(vis_save_path_1)

        fig = plt.gcf()
        plt.imshow(sig_src_2)
        plt.show()
        fig.savefig(vis_save_path_2)

def vis_src_sem(base_path, vis_img, vis=False):    
    vis_src_sem_dir = os.path.join(base_path, "vis_src_sem")
    make_dir(vis_src_sem_dir)
    src_sem_image = vis_img

    print("src_sem_image: ", src_sem_image.shape)         # (4, 128, 416, 6)

    for i in range(src_sem_image.shape[0]):
        sig_src_1 = src_sem_image[i, :, :, :3] 
        # print(sig_src_1.shape) # shape (128, 416, 3)
        sig_src_2 = src_sem_image[i, :, :, 3:6] 
        # print(sig_src_2.shape) # shape (128, 416, 3)
        vis_save_path_1 = os.path.join(vis_src_sem_dir, os.path.splitext(file_name)[0]+"-"+str(i)+"-1.png")
        vis_save_path_2 = os.path.join(vis_src_sem_dir, os.path.splitext(file_name)[0]+"-"+str(i)+"-2.png")

        fig = plt.gcf()
        plt.imshow(sig_src_1)
        plt.show()
        fig.savefig(vis_save_path_1)

        fig = plt.gcf()
        plt.imshow(sig_src_2)
        plt.show()
        fig.savefig(vis_save_path_2)

    
if __name__ == "__main__":
    base_path = os.path.join("/userhome/34/h3567721/projects/Depth/GeoNet/log/", "depth_geo_delta_two_stage_mask_fix_pose_vis_2")

    tgt_path = os.path.join(base_path, "tgt_image")
    src_path = os.path.join(base_path, "src_image_stack")

    tgt_sem_path = os.path.join(base_path, "tgt_sem")
    src_sem_path = os.path.join(base_path, "src_sem_stack")

    delta_path = os.path.join(base_path, "delta")
    depth_path = os.path.join(base_path, "depth")
    pose_path = os.path.join(base_path, "pose")

    # choose which layer will be visualized
    vis_var = ["depth", "delta", "pose", "tgt", "src", "tgt_sem", "src_sem"]
    # vis_var = ['delta']

    # depth img will always be visualized
    for file_name in os.listdir(depth_path):
        print(file_name)
        if "depth" in vis_var:
            depth_img = np.load(os.path.join(depth_path, file_name))
            vis_depth(base_path, vis_img=depth_img, vis=True)
        if "delta" in vis_var:
            delta_img = np.load(os.path.join(delta_path, file_name))
            vis_delta(base_path, vis_img=delta_img, vis=True)
        if "pose" in vis_var:
            pose_file = np.load(os.path.join(pose_path, file_name))
            vis_pose(base_path, vis_img=pose_file, vis=True)
        if "tgt" in vis_var:
            tgt_img = np.load(os.path.join(tgt_path, file_name))
            vis_tgt(base_path, vis_img=tgt_img, vis=True)
        if "src" in vis_var:
            src_img = np.load(os.path.join(src_path, file_name))
            vis_src(base_path, vis_img=src_img, vis=True)
        if "tgt_sem" in vis_var:
            tgt_sem_img = np.load(os.path.join(tgt_sem_path, file_name))
            vis_tgt_sem(base_path, vis_img=tgt_sem_img, vis=True)
        if "src_sem" in vis_var:
            src_sem_img = np.load(os.path.join(src_sem_path, file_name))
            vis_src_sem(base_path, vis_img=src_sem_img, vis=True)