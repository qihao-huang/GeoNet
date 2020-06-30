from __future__ import division
import tensorflow as tf
import numpy as np
import os
import PIL.Image as pil
from geonet_model import *


def test_depth(opt):
    ##### load testing list #####
    with open('data/kitti/test_files_%s.txt' % opt.depth_test_split, 'r') as f:
        test_files = f.readlines()
        test_files = [opt.dataset_dir + t[:-1] for t in test_files]
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    # print("all test_files: ", test_files)
    print("test_files: ", len(test_files))

    ##### init #####
    input_uint8 = tf.placeholder(tf.uint8, [opt.batch_size,
                                            opt.img_height, opt.img_width, 3], name='raw_input')

    # GeoNetModel(opt, tgt_image, src_image_stack, intrinsics):
    model = GeoNetModel(opt, input_uint8, None, None)
    fetches = {"depth": model.pred_depth[0]}

    saver = tf.train.Saver([var for var in tf.model_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ##### Go #####
    with tf.Session(config=config) as sess:
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
            fetcg_bwd_rigid_flow = []

        for t in range(0, len(test_files), opt.batch_size):
            if t % 100 == 0:
                print('processing: %d/%d' % (t, len(test_files)))
            inputs = np.zeros(
                (opt.batch_size, opt.img_height, opt.img_width, 3),
                dtype=np.uint8)

            for b in range(opt.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                fh = open(test_files[idx], 'r')
                raw_im = pil.open(fh)
                scaled_im = raw_im.resize(
                    (opt.img_width, opt.img_height), pil.ANTIALIAS)
                inputs[b] = np.array(scaled_im)

            if opt.save_test_intermediate:
                fetches["tgt_image"] = model.tgt_image # fetch tgt_image
                fetches["src_image_stack"] = model.src_image_stack # fetch src_image_stack    
                fetches["delta_xyz"] = model.delta_xyz[0] # fetch delta
                fetches["fwd_rigid_warp"] = model.fwd_rigid_warp_pyramid[0]
                fetches["bwd_rigid_warp"] = model.bwd_rigid_warp_pyramid[0]
                fetches["fwd_rigid_error"] = model.fwd_rigid_error_pyramid[0]
                fetches["bwd_rigid_error"] = model.bwd_rigid_error_pyramid[0]
                fetches["fetch_fwd_rigid_flow"] = model.fwd_rigid_flow_pyramid[0]
                fetches["fetcg_bwd_rigid_flow"] = model.bwd_rigid_flow_pyramid[0]

            pred = sess.run(fetches, feed_dict={input_uint8: inputs})
            for b in range(opt.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                pred_all.append(pred['depth'][b, :, :, 0])

                if opt.save_test_intermediate:
                    fetch_tgt_image.append(pred['tgt_image'][b, :, :, :]) # (128, 416, 3)
                    fetch_src_image_stack.append(pred['src_image_stack'][b, :, :, :]) # (128, 416, 6)
                    fetch_fwd_rigid_warp.append(pred['fwd_rigid_warp'][b, :, :, :])
                    fetch_bwd_rigid_warp.append(pred['bwd_rigid_warp'][b, :, :, :])
                    fetch_fwd_rigid_error.append(pred['fwd_rigid_error'][b, :, :, :])
                    fetch_bwd_rigid_error.append(pred['bwd_rigid_error'][b, :, :, :])
                    fetch_fwd_rigid_flow.append(fetches["fetch_fwd_rigid_flow"])
                    fetcg_bwd_rigid_flow.append(fetches["fetcg_bwd_rigid_flow"])
                                    

        np.save(opt.output_dir + '/' +
                os.path.basename(opt.init_ckpt_file), pred_all)

        # np: (697, 128, 416)
        if opt.save_test_intermediate:
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-tgt_image"), fetch_tgt_image)
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-src_image_stack"), fetch_src_image_stack)
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-delta_xyz"), fetch_delta_xyz)
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-fwd_rigid_warp"), fetch_fwd_rigid_warp)
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-bwd_rigid_warp"), fetch_bwd_rigid_warp)
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-fwd_rigid_error"), fetch_fwd_rigid_error)
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-bwd_rigid_error"), fetch_bwd_rigid_error)
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-fwd_rigid_flow"), fetch_fwd_rigid_flow)
            np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)+"-bwd_rigid_flow"), fetcg_bwd_rigid_flow)
