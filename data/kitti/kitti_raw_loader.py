# Mostly based on the code written by Tinghui Zhou: 
# https://github.com/tinghuiz/SfMLearner/blob/master/data/kitti/kitti_raw_loader.py
from __future__ import division
import numpy as np
from glob import glob
import os
import scipy.misc

class kitti_raw_loader(object):
    def __init__(self, 
                 dataset_dir,
                 split,
                 img_height=128,
                 img_width=416,
                 seq_length=3,
                 remove_static=True):
        # '~/projects/GeoNet/data/kitti'
        dir_path = os.path.dirname(os.path.realpath(__file__)) 

        # split: stereo or eigen
        # '~/projects/GeoNet/data/kitti/test_scenes_eigen.txt'
        # e.g 2011_09_26_drive_0059, 2011_09_26_drive_0117
        test_scene_file = dir_path + '/test_scenes_' + split + '.txt'

        with open(test_scene_file, 'r') as f:
            test_scenes = f.readlines()
        
        self.test_scenes = [t[:-1] for t in test_scenes]
        self.dataset_dir = dataset_dir
        self.img_height = img_height 
        self.img_width = img_width
        self.seq_length = seq_length

        # use both RGB cameras, left and right
        self.cam_ids = ['02', '03'] 
        self.date_list = ['2011_09_26', '2011_09_28', '2011_09_29', 
                          '2011_09_30', '2011_10_03']

        # read static frames from local static_frames.txt
        if remove_static:
            static_frames_file = dir_path + '/static_frames.txt'
            self.collect_static_frames(static_frames_file)

        self.collect_train_frames(remove_static)

        # self.static_frames[0]: 2011_09_26_drive_0009_sync 02 0000000386        
        # self.train_frames[0]: 2011_09_26_drive_0113_sync 02 0000000000

    def collect_static_frames(self, static_frames_file):
        with open(static_frames_file, 'r') as f:
            frames = f.readlines()
        self.static_frames = []
        for fr in frames:
            if fr == '\n':
                continue

            # date:2011_09_26 
            # drive: 2011_09_26_drive_0009_sync 
            # frame_id: 0000000394
            date, drive, frame_id = fr.split(' ')
            curr_fid = '%.10d' % (np.int(frame_id[:-1])) # 0000000394 

            # generate both cameras' static frames
            for cid in self.cam_ids:
                self.static_frames.append(drive + ' ' + cid + ' ' + curr_fid)
        
    def collect_train_frames(self, remove_static):
        all_frames = []
        for date in self.date_list:
            drive_set = os.listdir(self.dataset_dir + date + '/')
            for dr in drive_set:
                drive_dir = os.path.join(self.dataset_dir, date, dr)
                if os.path.isdir(drive_dir):
                    if dr[:-5] in self.test_scenes:
                        continue
                    for cam in self.cam_ids:
                        img_dir = os.path.join(drive_dir, 'image_' + cam, 'data')
                        N = len(glob(img_dir + '/*.png'))
                        for n in range(N):
                            frame_id = '%.10d' % n
                            all_frames.append(dr + ' ' + cam + ' ' + frame_id)
                        
        if remove_static:
            for s in self.static_frames:
                try:
                    all_frames.remove(s)
                except:
                    pass

        self.train_frames = all_frames
        self.num_train = len(self.train_frames)

    # ---------------------------------------------

    # function called by prepare_train_data
    def get_train_example_with_idx(self, tgt_idx):
        # only those frames are valid, otherwise return False
        if not self.is_valid_sample(self.train_frames, tgt_idx):
            return False
        example = self.load_example(self.train_frames, tgt_idx)

        return example

    def is_valid_sample(self, frames, tgt_idx):
        N = len(frames)
        tgt_drive, cid, _ = frames[tgt_idx].split(' ')
        half_offset = int((self.seq_length - 1)/2)
        min_src_idx = tgt_idx - half_offset
        max_src_idx = tgt_idx + half_offset
        if min_src_idx < 0 or max_src_idx >= N:
            return False

        min_src_drive, min_src_cid, _ = frames[min_src_idx].split(' ')
        max_src_drive, max_src_cid, _ = frames[max_src_idx].split(' ')
        if tgt_drive == min_src_drive and tgt_drive == max_src_drive and cid == min_src_cid and cid == max_src_cid:
            return True

        return False

    def load_example(self, frames, tgt_idx):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(frames, tgt_idx, self.seq_length)
        # zoom_x: 0.33494363929146537, 416/1242
        # zoom_y: 0.3413333333333333, 128/375

        # 2011_09_26_drive_0113_sync 02 0000000001
        # tgt_drive: 2011_09_26_drive_0113_sync, target drive
        # tgt_cid: 02, target camera id
        # tgt_frame_id: 0000000001, target frame it
        tgt_drive, tgt_cid, tgt_frame_id = frames[tgt_idx].split(' ')

        intrinsics = self.load_intrinsics_raw(tgt_drive, tgt_cid, tgt_frame_id)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        # array([[241.67446312,   0.        , 204.16801031],
        #         [  0.        , 246.28486827,  59.000832  ],
        #         [  0.        ,   0.        ,   1.        ]]

        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq # [(128, 416, 3), (128, 416, 3), (128, 416, 3)]
        example['folder_name'] = tgt_drive + '_' + tgt_cid + '/' # 2011_09_26_drive_0113_sync_02/
        example['file_name'] = tgt_frame_id # 0000000001

        return example

    def load_image_sequence(self, frames, tgt_idx, seq_length):
        half_offset = int((seq_length - 1)/2)
        image_seq = []
        for o in range(-half_offset, half_offset + 1):
            curr_idx = tgt_idx + o
            curr_drive, curr_cid, curr_frame_id = frames[curr_idx].split(' ')
            curr_img = self.load_image_raw(curr_drive, curr_cid, curr_frame_id)
            if o == 0:
                zoom_y = self.img_height/curr_img.shape[0]
                zoom_x = self.img_width/curr_img.shape[1]
            # resize the raw imgs
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
            image_seq.append(curr_img)

        return image_seq, zoom_x, zoom_y

    def load_image_raw(self, drive, cid, frame_id):
        date = drive[:10]
        img_file = os.path.join(self.dataset_dir, date, drive, 'image_' + cid, 'data', frame_id + '.png')
        img = scipy.misc.imread(img_file)
        return img

    def load_intrinsics_raw(self, drive, cid, frame_id):
        date = drive[:10]
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
