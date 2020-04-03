import os
import numpy as np

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    base_path = "/userhome/34/h3567721/projects/Depth/GeoNet/log/depth_geo_delta_vis"
    depth_path = os.path.join(base_path, "depth")
    delta_path = os.path.join(base_path, "delta")
    pose_path = os.path.join(base_path, "pose")

    vis_depth_dir = os.path.join(base_path, "vis_depth")
    if not os.path.exists(vis_depth_dir):
        os.makedirs(vis_depth_dir)

    vis_delta_dir = os.path.join(base_path, "vis_delta")
    if not os.path.exists(vis_delta_dir):
        os.makedirs(vis_delta_dir)

    vis_pose_dir = os.path.join(base_path, "vis_pose")
    if not os.path.exists(vis_pose_dir):
        os.makedirs(vis_pose_dir)

    for file_name in os.listdir(depth_path):
        print(file_name)
        depth_img = np.load(os.path.join(depth_path, file_name))
        pose_file = np.load(os.path.join(pose_path, file_name))
        # delta_img = np.load(os.path.join(delta_path, file_name))

        print("pose_file: ", pose_file.shape)

        for i in range(pose_file.shape[0]):
            sig_pose_1 = pose_file[i,0,:]
            sig_pose_2 = pose_file[i,1,:]
            vis_save_path_1 = os.path.join(vis_pose_dir, os.path.splitext(file_name)[0]+"-"+str(i)+"-1.txt")
            vis_save_path_2 = os.path.join(vis_pose_dir, os.path.splitext(file_name)[0]+"-"+str(i)+"-2.txt")
            np.savetxt(vis_save_path_1, sig_pose_1, delimiter=',')
            np.savetxt(vis_save_path_2, sig_pose_2, delimiter=',')

        print("depth_img: ", depth_img.shape)         # (12, 128, 416, 1)
        # print("delta_img: ", delta_img.shape)         # (4, 128, 416, 12)

        # for deltax_xyz, assumption: 
        # i=0: 0:3 -> fwd, 6:9 -> bwd
        # i=1: 3:6 -> fwd, 9:12 -> bwd

        for i in range(depth_img.shape[0]):
            sig_depth = depth_img[i, :, :, 0] 
            print(sig_depth.shape) # shape (128, 416)
            vis_save_path = os.path.join(base_path, "vis_depth", os.path.splitext(file_name)[0]+"-"+str(i)+".png")

            fig = plt.gcf()
            plt.imshow(sig_depth)
            plt.show()
            fig.savefig(vis_save_path)

# import numpy as np
# file_path = "/userhome/34/h3567721/projects/Depth/GeoNet/predictions/test_depth/model-240000.npy"
# depth_file = np.load(file_path)
# sig_depth = depth_file[0,:,:]