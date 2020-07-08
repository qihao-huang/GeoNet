from __future__ import division
import tensorflow as tf
import numpy as np
import os
import PIL.Image as pil
from geonet_model import *

def test_depth_delta(opt):
    # NOTE: only tested with eigen split
    ##### load testing list #####
    with open('data/kitti/test_files_%s.txt' % opt.depth_test_split, 'r') as f:
        test_files = f.readlines()
        test_files = [opt.dataset_dir + t[:-1] for t in test_files]

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    ##### init #####
    input_uint8_tgt = tf.compat.v1.placeholder(tf.uint8, [opt.batch_size,
                                            opt.img_height, opt.img_width, 3], name='raw_tgt_input')

    input_uint8_src = tf.compat.v1.placeholder(tf.uint8, [opt.batch_size,
                                            opt.img_height, opt.img_width, 6], name='raw_src_input')

    # GeoNetModel(opt, tgt_image, src_image_stack, intrinsics):
    model = GeoNetModel(opt, input_uint8_tgt, input_uint8_src, None)
    fetches = {"depth": model.pred_depth[0]} # (3, 128, 416, 1)

    saver = tf.compat.v1.train.Saver([var for var in tf.compat.v1.model_variables()])
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    ##### Go #####
    with tf.compat.v1.Session(config=config) as sess:
        saver.restore(sess, opt.init_ckpt_file)
        pred_all = []

        for t in range(0, len(test_files), opt.batch_size):
            if t % 100 == 0:
                print('processing: %d/%d' % (t, len(test_files)))
            
            inputs_tgt = np.zeros(
                (opt.batch_size, opt.img_height, opt.img_width, 3),
                dtype=np.uint8)

            inputs_src = np.zeros(
                (opt.batch_size, opt.img_height, opt.img_width, 6),
                dtype=np.uint8)

            # potential bug, only test with batch_size=1
            for b in range(opt.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break

                try:
                    # path/to/2011_09_26/2011_09_26_drive_0036_sync/image_02/data/0000000608.png
                    file_path = test_files[idx]
                    img_dir_path, img_full_name = os.path.split(file_path)
                    # focus on 3 frames first
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
                    # may fail if the idx < 0 or exceeds than range
                    # then print error to skip that tgt frame

                    raw_im_tgt = pil.open(fh_tgt)
                    raw_im_src_1 = pil.open(fh_src_1)
                    raw_im_src_2 = pil.open(fh_src_2)

                    scaled_im_tgt = raw_im_tgt.resize((opt.img_width, opt.img_height), pil.ANTIALIAS)
                    scaled_im_src_1 = raw_im_src_1.resize((opt.img_width, opt.img_height), pil.ANTIALIAS)
                    scaled_im_src_2 = raw_im_src_2.resize((opt.img_width, opt.img_height), pil.ANTIALIAS)

                    scaled_im_src_concat = np.concatenate((scaled_im_src_1, scaled_im_src_2), axis=2)

                    inputs_tgt[b] = np.array(scaled_im_tgt)
                    inputs_src[b] = np.array(scaled_im_src_concat)

                except Exception as e:
                    print(e)

            pred = sess.run(fetches, feed_dict={input_uint8_tgt: inputs_tgt, input_uint8_src: inputs_src})
            # feed_dict: __init__(self, opt, tgt_image, src_image_stack, intrinsics)

            # NOTE: only test with batch_size=1, b=0
            # So don't worry about this function
            for b in range(opt.batch_size): 
                idx = t + b
                if idx >= len(test_files):
                    break
                fetch_tgt_depth = pred['depth'][b, :, :, 0] #fetch target only
                pred_all.append(fetch_tgt_depth)

        # npy file will be saved locally
        np.save(os.path.join(opt.output_dir, os.path.basename(opt.init_ckpt_file)), pred_all)