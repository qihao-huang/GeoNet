from __future__ import division
import tensorflow as tf
import numpy as np
import os
import PIL.Image as pil
from geonet_model import *

def scale_intrinsics(mat, sx, sy):
    # K = | fx   0   cx| = | fx*scale_x   0            cx*scale_x|
    #     |  0  fy   cy|   |  0           fy*scale_y   cy*scale_y|
    #     |  0   0    1|   |  0           0            1         |
    out = np.copy(mat)
    out[0,0] *= sx
    out[0,2] *= sx
    out[1,1] *= sy
    out[1,2] *= sy

    return out

def read_raw_calib_file(filepath):
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
    
def make_intrinsics_matrix(fx, fy, cx, cy):
    # Assumes batch input
    # K = | fx   0   cx|
    #     |  0  fy   cy|
    #     |  0   0    1|

    r1 = np.array([fx, 0., cx])
    r2 = np.array([0., fy, cy])
    r3 = np.array([0., 0., 1.])
    intrinsics = np.stack([r1, r2, r3], axis=1)  # (3,3)

    return intrinsics

def load_intrinsics_raw(opt, drive, cid, frame_id):
    date = drive[:10]
    calib_file = os.path.join(opt.dataset_dir, date, 'calib_cam_to_cam.txt')

    filedata = read_raw_calib_file(calib_file)
    P_rect = np.reshape(filedata['P_rect_' + cid], (3, 4))
    # array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
    #        [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
    #        [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]]))

    intrinsics = P_rect[:3, :3]

    return intrinsics

def get_multi_scale_intrinsics(intrinsics, num_scales):
    # num_scales = 4
    intrinsics_mscale = []
    # Scale the intrinsics accordingly for each scale
    # 2**0=1, 2**1=2, 2**3=8, 2**4=16
    for s in range(num_scales):
        fx = intrinsics[0, 0]/(2 ** s)
        fy = intrinsics[1, 1]/(2 ** s)
        cx = intrinsics[0, 2]/(2 ** s)
        cy = intrinsics[1, 2]/(2 ** s)

        intrinsics_mscale.append(make_intrinsics_matrix(fx, fy, cx, cy))

    intrinsics_mscale = np.stack(intrinsics_mscale, axis=0)

    return intrinsics_mscale

def test_depth_delta(opt):
    # NOTE: only tested with eigen split
    ##### load testing list #####
    with open('data/kitti/test_files_%s.txt' % opt.depth_test_split, 'r') as f:
        test_files = f.readlines()
        test_files = [opt.dataset_dir + t[:-1] for t in test_files]

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    ##### init #####
    input_uint8_tgt = tf.compat.v1.placeholder(tf.uint8, [opt.batch_size, opt.img_height, opt.img_width, 3], name='raw_tgt_input')
    input_uint8_src = tf.compat.v1.placeholder(tf.uint8, [opt.batch_size, opt.img_height, opt.img_width, 6], name='raw_src_input')
    input_float32_src = tf.compat.v1.placeholder(tf.float32, [opt.batch_size, opt.num_scales, 3, 3], name='raw_intrinsic_input')

    # GeoNetModel(opt, tgt_image, src_image_stack, intrinsics):
    model = GeoNetModel(opt, input_uint8_tgt, input_uint8_src, input_float32_src)
    fetches = {"depth": model.pred_depth[0]} # (3, 128, 416, 1)

    saver = tf.compat.v1.train.Saver([var for var in tf.compat.v1.model_variables()])
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    ##### Go #####
    with tf.compat.v1.Session(config=config) as sess:
        saver.restore(sess, opt.init_ckpt_file)
        pred_all = []
        
        if opt.save_test_intermediate:
            fetch_tgt_image = []
            fetch_src_image_stack = []
            fetch_delta_xyz = []
            fetch_fwd_rigid_warp = []
            fetch_bwd_rigid_warp = []
            fetch_fwd_rigid_error = []
            fetch_bwd_rigid_error = []
            fetch_fwd_rigid_flow = []
            fetch_bwd_rigid_flow = []

        for t in range(0, len(test_files), opt.batch_size):
            if t % 100 == 0:
                print('processing: %d/%d' % (t, len(test_files)))
            
            inputs_tgt = np.zeros((opt.batch_size, opt.img_height, opt.img_width, 3), dtype=np.uint8)
            inputs_src = np.zeros((opt.batch_size, opt.img_height, opt.img_width, 6), dtype=np.uint8)
            inputs_intrinsic = np.zeros((opt.batch_size, opt.num_scales, 3, 3), dtype=np.float32)

            # only test with batch_size=1
            for b in range(opt.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break

                # path/to/2011_09_26/2011_09_26_drive_0036_sync/image_02/data/0000000608.png
                file_path = test_files[idx]
                img_dir_path, img_full_name = os.path.split(file_path)
                img_name = os.path.splitext(img_full_name)[0] # '0000000608'
                img_tgt_idx = int(img_name)

                if img_tgt_idx == 0: #"0000000000"
                    img_src_1_idx = int(img_name) #"0000000000"
                    img_src_2_idx = int(img_name)+1 #"00000000001"
                elif int(img_name)+1 >= len(os.listdir(img_dir_path)):
                    # /userhome/34/h3567721/dataset/kitti/raw_data/
                    # 2011_09_26/2011_09_26_drive_0059_sync/image_02/data/
                    # 0000000372.png
                    # There is no 0000000373 image
                    img_src_1_idx = int(img_name)-1 # '0000000371'
                    img_src_2_idx = int(img_name)   # '0000000372'
                else:
                    img_src_1_idx = int(img_name)-1 # '0000000607'
                    img_src_2_idx = int(img_name)+1 # '0000000609'
                    
                img_tgt_idx_path = os.path.join(img_dir_path, str(img_tgt_idx).zfill(len(img_name))+".png")
                img_src_1_idx_path = os.path.join(img_dir_path, str(img_src_1_idx).zfill(len(img_name))+".png")
                img_src_2_idx_path = os.path.join(img_dir_path, str(img_src_2_idx).zfill(len(img_name))+".png")

                fh_tgt = open(img_tgt_idx_path, 'r')
                fh_src_1 = open(img_src_1_idx_path, 'r')
                fh_src_2 = open(img_src_2_idx_path, 'r')

                raw_im_tgt = pil.open(fh_tgt)
                raw_im_src_1 = pil.open(fh_src_1)
                raw_im_src_2 = pil.open(fh_src_2)

                scaled_im_tgt = raw_im_tgt.resize((opt.img_width, opt.img_height), pil.ANTIALIAS)
                scaled_im_src_1 = raw_im_src_1.resize((opt.img_width, opt.img_height), pil.ANTIALIAS)
                scaled_im_src_2 = raw_im_src_2.resize((opt.img_width, opt.img_height), pil.ANTIALIAS)

                scaled_im_src_concat = np.concatenate((scaled_im_src_1, scaled_im_src_2), axis=2)

                zoom_x = opt.img_width/scaled_im_tgt.size[0] # 416/1242
                zoom_y = opt.img_height/scaled_im_tgt.size[1] # 128/375

                tgt_drive = img_dir_path.split("/")[-3]
                tgt_cid = img_dir_path.split("/")[-2].split("_")[-1]
                tgt_frame_id = img_name
                intrinsics = load_intrinsics_raw(opt, tgt_drive, tgt_cid, tgt_frame_id)
                intrinsics = scale_intrinsics(intrinsics, zoom_x, zoom_y)
                intrinsics = get_multi_scale_intrinsics(intrinsics, opt.num_scales)

                inputs_tgt[b] = np.array(scaled_im_tgt)
                inputs_src[b] = np.array(scaled_im_src_concat)
                # intrinsic of src images and tgt image are same in one scene event
                inputs_intrinsic[b] = intrinsics # (4,3,3)

            if opt.save_test_intermediate:
                fetches["tgt_image"] = model.tgt_image # fetch tgt_image
                fetches["src_image_stack"] = model.src_image_stack # fetch src_image_stack    
                fetches["delta_xyz"] = model.delta_xyz[0] # fetch delta
                fetches["fwd_rigid_warp"] = model.fwd_rigid_warp_pyramid[0]
                fetches["bwd_rigid_warp"] = model.bwd_rigid_warp_pyramid[0]
                fetches["fwd_rigid_error"] = model.fwd_rigid_error_pyramid[0]
                fetches["bwd_rigid_error"] = model.bwd_rigid_error_pyramid[0]
                fetches["fwd_rigid_flow"] = model.fwd_rigid_flow_pyramid[0]
                fetches["bwd_rigid_flow"] = model.bwd_rigid_flow_pyramid[0]
            
            pred = sess.run(fetches, feed_dict={input_uint8_tgt: inputs_tgt, input_uint8_src: inputs_src, input_float32_src: inputs_intrinsic})

            # NOTE: only test with batch_size=1, so b=0
            for b in range(opt.batch_size): 
                idx = t + b
                if idx >= len(test_files):
                    break

                fetch_tgt_depth = pred['depth'][b, :, :, 0] #fetch target only
                pred_all.append(fetch_tgt_depth)

                if opt.save_test_intermediate:
                    fetch_tgt_image.append(pred['tgt_image'][b, :, :, :]) # (128, 416, 3)
                    fetch_src_image_stack.append(pred['src_image_stack'][b, :, :, :]) # (128, 416, 6)
                    fetch_delta_xyz.append(pred['delta_xyz'][b, :, :, :]) # (128, 416, 12)
                    fetch_fwd_rigid_warp.append(pred['fwd_rigid_warp'][b, :, :, :])
                    fetch_bwd_rigid_warp.append(pred['bwd_rigid_warp'][b, :, :, :])
                    fetch_fwd_rigid_error.append(pred['fwd_rigid_error'][b, :, :, :])
                    fetch_bwd_rigid_error.append(pred['bwd_rigid_error'][b, :, :, :])
                    fetch_fwd_rigid_flow.append(fetches["fwd_rigid_flow"][b, :, :, :])
                    fetch_bwd_rigid_flow.append(fetches["bwd_rigid_flow"][b, :, :, :])
                                    
        # npy file will be saved locally
        np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)), pred_all)

        if opt.save_test_intermediate:
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-tgt_image"), fetch_tgt_image)
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-src_image_stack"), fetch_src_image_stack)
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-delta_xyz"), fetch_delta_xyz)
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-fwd_rigid_warp"), fetch_fwd_rigid_warp)
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-bwd_rigid_warp"), fetch_bwd_rigid_warp)
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-fwd_rigid_error"), fetch_fwd_rigid_error)
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-bwd_rigid_error"), fetch_bwd_rigid_error)
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-fwd_rigid_flow"), fetch_fwd_rigid_flow)
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-bwd_rigid_flow"), fetch_bwd_rigid_flow)

