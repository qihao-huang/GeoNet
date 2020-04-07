from __future__ import division
import tensorflow as tf
import numpy as np
import os
import PIL.Image as pil
from geonet_model import *


def test_depth_delta(opt):
    ##### load testing list #####
    with open('data/kitti/test_files_%s.txt' % opt.depth_test_split, 'r') as f:
        test_files = f.readlines()
        test_files = [opt.dataset_dir + t[:-1] for t in test_files]
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    print("test_files: ", len(test_files))

    ##### init #####
    # NOTE: concat 3 consecutive frames into one to match the delta_xyz training
    # only test with batch_size = 1, (3 (tgt, src_1, src_2), 128, 416, 3)
    input_uint8 = tf.placeholder(tf.uint8, [opt.batch_size*3,
                                            opt.img_height, opt.img_width, 3], name='raw_input')

    # GeoNetModel(opt, tgt_image, src_image_stack, intrinsics):
    model = GeoNetModel(opt, input_uint8, None, None)
    fetches = {"depth": model.pred_depth[0]} # (3, 128, 416, 1)

    saver = tf.train.Saver([var for var in tf.model_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ##### Go #####
    with tf.Session(config=config) as sess:
        saver.restore(sess, opt.init_ckpt_file)
        pred_all = []
        for t in range(0, len(test_files), opt.batch_size):
            if t % 100 == 0:
                print('processing: %d/%d' % (t, len(test_files)))
            
            inputs = np.zeros(
                (opt.batch_size*3, opt.img_height, opt.img_width, 3),
                dtype=np.uint8)

            # potential bug, only test with batch_size=1
            for b in range(opt.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break

                try:
                    # path/to/2011_09_26/2011_09_26_drive_0036_sync/image_02/data/0000000608.png
                    file_path = test_files[idx]
                    img_name = os.path.basename(file_path)

                    # focus on 3 frames first
                    img_tgt_idx = int(os.path.splitext(img_name)[0])
                    img_src_1_idx = int(os.path.splitext(img_name)[0])-1
                    img_src_2_idx = int(os.path.splitext(img_name)[0])+1
                    print("img_tgt_idx: ", img_tgt_idx)
                    print("img_src_1_idx: ", img_src_1_idx)
                    print("img_src_2_idx: ", img_src_2_idx)

                    fh_tgt = open(img_tgt_idx, 'r')
                    fh_src_1 = open(img_src_1_idx, 'r')
                    fh_src_2 = open(img_src_2_idx, 'r')
                    # may fail if the idx < 0 or bigger than range
                    # then print error to skip that tgt frame

                    raw_im_tgt = pil.open(fh_tgt)
                    raw_im_src_1 = pil.open(fh_src_1)
                    raw_im_src_2 = pil.open(fh_src_2)

                    scaled_im_tgt = raw_im_tgt.resize((opt.img_width, opt.img_height), pil.ANTIALIAS)
                    scaled_im_src_1 = raw_im_src_1.resize((opt.img_width, opt.img_height), pil.ANTIALIAS)
                    scaled_im_src_2 = raw_im_src_2.resize((opt.img_width, opt.img_height), pil.ANTIALIAS)

                    # tf.concat([self.dispnet_inputs, self.src_image_stack[:,:,:,3*i:3*(i+1)]], axis=0)
                    tmp = np.concatenate((scaled_im_tgt, scaled_im_src_1), axis=0)
                    scaled_im_concat = np.concatenate((tmp, scaled_im_src_2), axis=0)
                    print("scaled_im_concat shape: ", scaled_im_concat.shape)

                    inputs[b] = np.array(scaled_im_concat)

                except Exception as e:
                    print("skip img: ", file_path)
                    print(e)

            pred = sess.run(fetches, feed_dict={input_uint8: inputs})
            for b in range(opt.batch_size): # NOTE: only test with batch_size=1, b=0
                idx = t + b
                if idx >= len(test_files):
                    break
                fetch_tgt_depth = pred['depth'][b, :, :, 0] #fetch target only
                print("fetch_tgt_depth shape: ", fetch_tgt_depth.shape)
                pred_all.append(fetch_tgt_depth)  

        np.save(opt.output_dir + '/' +
                os.path.basename(opt.init_ckpt_file), pred_all)
