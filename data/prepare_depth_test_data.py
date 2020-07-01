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
parser.add_argument("--testset_file",  type=str, required=True, help="files point to the test images")
parser.add_argument("--dump_root",     type=str, required=True, help="where to dump the data")
parser.add_argument("--seq_length",    type=int, required=True, help="length of each training sequence")
parser.add_argument("--img_height",    type=int, default=128,   help="image height")
parser.add_argument("--img_width",     type=int, default=416,   help="image width")
parser.add_argument("--num_threads",   type=int, default=4,     help="number of threads to use")
args = parser.parse_args()

# python data/prepare_depth_test_data.py \
#     --dataset_dir="/userhome/34/h3567721/dataset/kitti/raw_data/" \
#     --testset_file="/userhome/34/h3567721/projects/Depth/GeoNet/data/kitti/test_files_eigen.txt" \
#     --dump_root="/userhome/34/h3567721/dataset/kitti/kitti_depth_test_eigen/" \
#     --seq_length=3 \
#     --img_height=128 \
#     --img_width=416 \
#     --num_threads=16

class kitti_test_loader(object):
    def __init__(self, 
                 dataset_dir,
                 testset_file,
                 split,
                 img_height=128,
                 img_width=416,
                 seq_length=3):

        with open(testset_file, 'r') as f:
            test_scenes = f.readlines()

        # "2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000069.png\n"
        self.test_frames = [test_path.replace("\n","") for test_path in test_scenes]
        self.num_train = len(self.test_frames)

        self.dataset_dir = dataset_dir
        self.img_height = img_height 
        self.img_width = img_width
        self.seq_length = seq_length

    # ---------------------------------------------

    def get_train_example_with_idx(self, tgt_idx):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(self.test_frames, tgt_idx, self.seq_length)
        # zoom_x: 0.33494363929146537, 416/1242
        # zoom_y: 0.3413333333333333, 128/375

        # tgt_drive: 2011_09_26_drive_0113_sync, target drive
        # tgt_cid: 02, target camera id
        # tgt_frame_id: 0000000001, target frame it
        tgt_drive = self.test_frames[tgt_idx].split('/')[1]
        tgt_cid = self.test_frames[tgt_idx].split('/')[2].split("_")[1]
        tgt_frame_id = self.test_frames[tgt_idx].split('/')[-1].split(".")[0]

        intrinsics = self.load_intrinsics_raw(tgt_drive, tgt_cid, tgt_frame_id)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)

        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq # [(128, 416, 3), (128, 416, 3), (128, 416, 3)]
        example['folder_name'] = tgt_drive + '_' + tgt_cid + '/' # 2011_09_26_drive_0113_sync_02/
        example['file_name'] = tgt_frame_id # 0000000001

        return example

    def load_image_sequence(self, frames, tgt_idx, seq_length):
        # NOTE: only test with seq_length=3

        # curr_drive: 2011_09_26_drive_0027_sync
        # curr_cir: 02
        # curr_frame_id: 0000000098
        curr_drive = frames[tgt_idx].split('/')[1]
        curr_cid = frames[tgt_idx].split('/')[2].split("_")[1]
        curr_frame_id = frames[tgt_idx].split('/')[-1].split(".")[0]

        img_dir_path = os.path.join(self.dataset_dir, os.path.split(frames[tgt_idx])[0])

        if int(curr_frame_id) == 0: #"0000000000"
            curr_frame_id_0 = int(curr_frame_id)    #"00000000000"
            curr_frame_id_1 = int(curr_frame_id)    #"00000000000"
            curr_frame_id_2 = int(curr_frame_id)+1  #"00000000001"
        elif int(curr_frame_id)+1 >= len(os.listdir(img_dir_path)):
            curr_frame_id_0 = int(curr_frame_id)-1   #"0000000371"
            curr_frame_id_1 = int(curr_frame_id)     #"0000000372"
            curr_frame_id_2 = int(curr_frame_id)     #"0000000372"
        else:
            curr_frame_id_0 = int(curr_frame_id)-1   #"0000000001"
            curr_frame_id_1 = int(curr_frame_id)     #"0000000002"
            curr_frame_id_2 = int(curr_frame_id)+1   #"0000000003"

        curr_img_0 = self.load_image_raw(curr_drive, curr_cid, str(curr_frame_id_0).zfill(len(curr_frame_id)))
        curr_img_1 = self.load_image_raw(curr_drive, curr_cid, str(curr_frame_id_1).zfill(len(curr_frame_id)))
        curr_img_2 = self.load_image_raw(curr_drive, curr_cid, str(curr_frame_id_2).zfill(len(curr_frame_id)))

        zoom_y = self.img_height/curr_img_1.shape[0]
        zoom_x = self.img_width/curr_img_1.shape[1]

        # resize the raw imgs
        curr_img_0 = scipy.misc.imresize(curr_img_0, (self.img_height, self.img_width))
        curr_img_1 = scipy.misc.imresize(curr_img_1, (self.img_height, self.img_width))
        curr_img_2 = scipy.misc.imresize(curr_img_2, (self.img_height, self.img_width))
        
        image_seq = [curr_img_0, curr_img_1, curr_img_2]

        return image_seq, zoom_x, zoom_y

    def load_image_raw(self, drive, cid, frame_id):
        date = drive[:10] # "2011_09_26"
        img_file = os.path.join(self.dataset_dir, date, drive, 'image_' + cid, 'data', frame_id + '.png')
        img = scipy.misc.imread(img_file)
        return img

    def load_intrinsics_raw(self, drive, cid, frame_id):
        date = drive[:10] # "2011_09_26"
        calib_file = os.path.join(self.dataset_dir, date, 'calib_cam_to_cam.txt')

        filedata = self.read_raw_calib_file(calib_file)
        P_rect = np.reshape(filedata['P_rect_' + cid], (3, 4))
        # array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
        #        [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
        #        [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]]))

        intrinsics = P_rect[:3, :3]

        return intrinsics
    
    def read_raw_calib_file(self,filepath):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                        data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                        pass
        return data

    def scale_intrinsics(self, mat, sx, sy):
        # K = | fx   0   cx| = | fx*scale_x   0            cx*scale_x|
        #     |  0  fy   cy|   |  0           fy*scale_y   cy*scale_y|
        #     |  0   0    1|   |  0           0            1         |
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy

        return out


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

    global data_loader

    # depth
    data_loader = kitti_test_loader(args.dataset_dir,
                                    args.testset_file,
                                    split='eigen',
                                    img_height=args.img_height,
                                    img_width=args.img_width,
                                    seq_length=args.seq_length)


    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, args) for n in range(data_loader.num_train))

main()
