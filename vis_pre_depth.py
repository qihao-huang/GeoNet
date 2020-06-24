import os
import numpy as np
import shutil
import cv2

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)

if __name__ == "__main__":
    # base_path = "/userhome/34/h3567721/projects/Depth/GeoNet/predictions"
    # dir_name = "test_xyz_depth_delta_two_stage"

    base_path = "/userhome/34/h3567721/projects/Depth/GeoNet/GeoNet_models_and_predictions/predictions"
    dir_name = "depth_result"

    file_name = "model_sn"

    depth_path = os.path.join(base_path, dir_name)
    vis_depth_dir = os.path.join(base_path, dir_name+"_vis", file_name)    
    make_dir(vis_depth_dir)
    depth_file = np.load(os.path.join(depth_path, file_name + ".npy"))

    print("depth_file: ", depth_file.shape)         # (12, 128, 416, 1)

    for i in range(depth_file.shape[0]):
        print(i)
        sig_depth = depth_file[i, :, :] 
        # print(sig_depth.shape) # shape (128, 416)
        vis_save_path = os.path.join(vis_depth_dir, str(i)+".png")

        fig = plt.gcf()

        plt.imshow(sig_depth, cmap='plasma')
        plt.axis("off")
        fig.set_size_inches(sig_depth.shape[1]/100.0, sig_depth.shape[0]/100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0,0)

        plt.show()
        fig.savefig(vis_save_path, dpi=96.0, pad_inches=0.0)