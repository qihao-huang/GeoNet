from __future__ import division
import tensorflow as tf
import numpy as np
import os
import PIL.Image as pil
from geonet_model import *
import time

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def make_intrinsics_matrix(fx, fy, cx, cy):
    r1 = [fx, 0., cx]
    r2 = [0., fy, cy]
    r3 = [0., 0., 1.]

    return np.array([r1,r2,r3])

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

    make_dir(os.path.join(opt.output_dir, "depth"))
    make_dir(os.path.join(opt.output_dir, "tgt_image"))
    make_dir(os.path.join(opt.output_dir, "src_image_stack"))
    make_dir(os.path.join(opt.output_dir, "delta_xyz"))
    make_dir(os.path.join(opt.output_dir, "fwd_rigid_warp"))
    make_dir(os.path.join(opt.output_dir, "bwd_rigid_warp"))
    make_dir(os.path.join(opt.output_dir, "fwd_rigid_error"))
    make_dir(os.path.join(opt.output_dir, "bwd_rigid_error"))
    make_dir(os.path.join(opt.output_dir, "fwd_rigid_flow"))
    make_dir(os.path.join(opt.output_dir, "bwd_rigid_flow"))

    ##### init #####
    input_uint8_tgt = tf.compat.v1.placeholder(tf.uint8, [opt.batch_size, opt.img_height, opt.img_width, 3], name='raw_tgt_input')
    input_uint8_src = tf.compat.v1.placeholder(tf.uint8, [opt.batch_size, opt.img_height, opt.img_width, 6], name='raw_src_input')
    input_float32_src = tf.compat.v1.placeholder(tf.float32, [opt.batch_size, opt.num_scales, 3, 3], name='raw_intrinsic_input')

    # GeoNetModel(opt, tgt_image, src_image_stack, intrinsics):
    model = GeoNetModel(opt, input_uint8_tgt, input_uint8_src, input_float32_src)
    fetches = {"depth": model.pred_depth[0]} # (3, 128, 416, 1) ,since bs=1, so is 3 in first channel

    saver = tf.compat.v1.train.Saver([var for var in tf.compat.v1.model_variables()])
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    ##### Go #####
    with tf.compat.v1.Session(config=config) as sess:
        saver.restore(sess, opt.init_ckpt_file)
        pred_all = []
        
        if opt.save_intermediate:
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

                # "2011_09_26/2011_09_26_drive_0036_sync/image_02/data/0000000608.png"
                file_path = test_files[idx]
                img_dir_path, img_full_name = os.path.split(file_path)
                img_name = os.path.splitext(img_full_name)[0] # '0000000608'

                img_path = os.path.join(img_dir_path, img_name+".jpg")
                cam_txt = os.path.join(img_dir_path, img_name+"_cam.txt")

                fh = open(img_path, 'r')
                raw_im = pil.open(fh)

                im_src_1 = np.array(raw_im.crop((0,0,opt.img_width ,opt.img_height)))
                im_tgt = np.array(raw_im.crop((opt.img_width, 0, opt.img_width *2, opt.img_height)))
                im_src_2 = np.array(raw_im.crop((opt.img_width *2, 0, opt.img_width *3, opt.img_height)))

                with open(cam_txt, 'r') as f:
                    intrinsirc_raw = f.readline()

                intrinsirc_mat = np.array([float(num) for num in intrinsirc_raw.split(",")])
    
                inputs_tgt[b] = im_tgt
                inputs_src[b] = np.concatenate([im_src_1, im_src_2], axis=2)
                inputs_intrinsic[b] = get_multi_scale_intrinsics(intrinsirc_mat, opt.num_scales)

                print("inputs_tgt shape: ", inputs_tgt[b])
                print("inputs_src shape: ", inputs_src[b])
                print("inputs_intrinsic shape: ", inputs_intrinsic[b])

            if opt.save_intermediate:
                fetches["tgt_image"] = model.tgt_image # fetch tgt_image
                fetches["src_image_stack"] = model.src_image_stack # fetch src_image_stack    
                fetches["delta_xyz"] = model.delta_xyz[0] # fetch delta
                fetches["fwd_rigid_warp"] = model.fwd_rigid_warp_pyramid[0]
                fetches["bwd_rigid_warp"] = model.bwd_rigid_warp_pyramid[0]
                fetches["fwd_rigid_error"] = model.fwd_rigid_error_pyramid[0]
                fetches["bwd_rigid_error"] = model.bwd_rigid_error_pyramid[0]
                fetches["fwd_rigid_flow"] = model.fwd_rigid_flow_pyramid[0]
                fetches["bwd_rigid_flow"] = model.bwd_rigid_flow_pyramid[0]
            
            # NOTE: save_intermediate: need intinsic to save error and warp
            pred = sess.run(fetches, feed_dict={input_uint8_tgt: inputs_tgt, input_uint8_src: inputs_src, input_float32_src: inputs_intrinsic})

            for b in range(opt.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                pred_all.append(pred['depth'][b, :, :, 0])
            
            if opt.save_intermediate:
                file_name = str(time.time())
                
                np.save(os.path.join(opt.output_dir, "depth", file_name), pred['depth'])
                np.save(os.path.join(opt.output_dir, "tgt_image", file_name), pred['tgt_image'])
                np.save(os.path.join(opt.output_dir, "src_image_stack", file_name), pred['src_image_stack'])
                np.save(os.path.join(opt.output_dir, "delta_xyz", file_name), pred['delta_xyz'])
                np.save(os.path.join(opt.output_dir, "fwd_rigid_warp", file_name), pred['fwd_rigid_warp'])
                np.save(os.path.join(opt.output_dir, "bwd_rigid_warp", file_name), pred['bwd_rigid_warp'])
                np.save(os.path.join(opt.output_dir, "fwd_rigid_error", file_name), pred['fwd_rigid_error'])
                np.save(os.path.join(opt.output_dir, "bwd_rigid_error", file_name), pred['bwd_rigid_error'])
                np.save(os.path.join(opt.output_dir, "fwd_rigid_flow", file_name), pred['fwd_rigid_flow'])
                np.save(os.path.join(opt.output_dir, "bwd_rigid_flow", file_name), pred['bwd_rigid_flow'])

        # npy file will be saved locally
        np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)), pred_all)