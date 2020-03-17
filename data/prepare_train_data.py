# -*- coding: utf-8 -*-
# Mostly based on the code written by Tinghui Zhou: 
# https://github.com/tinghuiz/SfMLearner/blob/master/data/prepare_train_data.py

from __future__ import division
import argparse
import scipy.misc
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",   type=str, required=True, help="where the dataset is stored")
parser.add_argument("--dataset_name",  type=str, required=True, choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes"])
parser.add_argument("--dump_root",     type=str, required=True, help="where to dump the data")
parser.add_argument("--seq_length",    type=int, required=True, help="length of each training sequence")
parser.add_argument("--img_height",    type=int, default=128,   help="image height")
parser.add_argument("--img_width",     type=int, default=416,   help="image width")
parser.add_argument("--num_threads",   type=int, default=4,     help="number of threads to use")
parser.add_argument("--remove_static", help="remove static frames from kitti raw data", action='store_true')
args = parser.parse_args()

def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res

def dump_example(n, args):
    if n % 2000 == 0:
        print('Progress %d/%d....' % (n, data_loader.num_train))
    
    example = data_loader.get_train_example_with_idx(n)
    if example == False:
        return
    
    # horizontally concat 3 or 5 frames
    image_seq = concat_image_seq(example['image_seq'])
    intrinsics = example['intrinsics']

    # Extrinsic Matrix
    # [R|t] = | r1,1  r2,1  r3,1  t1 |
    #         | r1,2  r2,2  r3,2  t2 |
    #         | r1,3  r2,3  r3,3  t3 ⎥
    #         |    0     0     0   1 |   
    
    # |R t| = |I t|   |R 0| =  | 1 0 0 t1 |     | r1,1  r2,1  r3,1 0 |
    # |0 1|   |0 1| × |0 1|    | 0 1 0 t2 |  x  | r1,2  r2,2  r3,2 0 |
    #                          | 0 0 1 t3 |     | r1,3  r2,3  r3,3 0 ⎥
    #                          | 0 0 0  1 |     |    0     0     0 1 ⎥ 
    #                          3D Transaltion       3D Rotation

    # Intrinsic Matrix
    # K = | fx   s   cx|
    #     |  0  fy   cy|
    #     |  0   0   1 |

    # K = |  1   0   cx|     | fx   0   0|     |  1   s/fx 0 |
    #     |  0   1   cy|  x  |  0  fy   0|  x  |  0   1    0 |
    #     |  0   0    1|     |  0   0   1|     |  0   0    1 |
    #     2D Translation     2D Scaling        2D shear

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    dump_dir = os.path.join(args.dump_root, example['folder_name'])

    try: 
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    
    # for task in [pose, depth, flow]:
    #   generate consecutive 3 or 5 horizontal concated new jpg image 
    #   and its corresponding camera pose txt file

    # 0000000276.jpg
    # 3 frames: 1248x128
    # 5 frames: 2080x128
    dump_img_file = dump_dir + '/%s.jpg' % example['file_name']
    scipy.misc.imsave(dump_img_file, image_seq.astype(np.uint8))

    # 0000000276_cam.txt
    # K = | fx   0   cx|
    #     |  0  fy   cy|
    #     |  0   0    1|

    # K = | 241.674463  0           204.168010 |
    #     | 0           246.284868  59.000832  |
    #     | 0           0           1          |
    # 241.674463,0.,204.168010,0.,246.284868,59.000832,0.,0.,1.
    dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
    with open(dump_cam_file, 'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))


def main():
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)

    # initialize the data_loader for each task
    global data_loader
    # pose
    if args.dataset_name == 'kitti_odom':
        from kitti.kitti_odom_loader import kitti_odom_loader
        data_loader = kitti_odom_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)
        print("hello")
        print(data_loader.num_train)
        print("hello babe")

    # depth
    if args.dataset_name == 'kitti_raw_eigen':
        from kitti.kitti_raw_loader import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='eigen',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length,
                                       remove_static=args.remove_static)

    # flow
    if args.dataset_name == 'kitti_raw_stereo':
        from kitti.kitti_raw_loader import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='stereo',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length,
                                       remove_static=args.remove_static)

    if args.dataset_name == 'cityscapes':
        from cityscapes.cityscapes_loader import cityscapes_loader
        data_loader = cityscapes_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)

    # multi-threads for data preprocessing 
    # 45016 for depth, raw_data, kitti_raw_eigen
    # 59552 for flow, raw_data, kitti_raw_stereo
    # 20409 for pose, odometry, kitti_odom
    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, args) for n in range(data_loader.num_train))

    # Split into train/val, random seed to make it consistent
    np.random.seed(8964)
    subfolders = os.listdir(args.dump_root)
    with open(os.path.join(args.dump_root, 'train.txt'), 'w') as tf:
        with open(os.path.join(args.dump_root, 'val.txt'), 'w') as vf:
            for s in subfolders:
                if not os.path.isdir(args.dump_root + '/%s' % s):
                    continue
                
                imfiles = glob(os.path.join(args.dump_root, s, '*.jpg'))
                frame_ids = [os.path.basename(fi).split('.')[0] for fi in imfiles]

                for frame in frame_ids:
                    # 10% for validation, 90% for training
                    if np.random.random() < 0.1:
                        vf.write('%s %s\n' % (s, frame))
                    else:
                        tf.write('%s %s\n' % (s, frame))

main()

