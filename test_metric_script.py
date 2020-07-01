from __future__ import division
import os

import time
import random
import pprint
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from geonet_model import *
from data_loader import DataLoader

import PIL.Image as pil

import argparse
from kitti_eval.depth_evaluation_utils import *

flags = tf.app.flags
####
flags.DEFINE_boolean("save_intermedia",           False, "whether to save the intermediate vars")
flags.DEFINE_boolean("delta_mode",                False, "whether to train the delta xyz")

####

flags.DEFINE_string("mode",                         "", "(train_rigid, train_flow) or (test_depth, test_pose, test_flow)")
flags.DEFINE_string("dataset_dir",                  "", "Dataset directory")
flags.DEFINE_string("ckpt_dir",                  None, "Specific checkpoint dir to eval all")
flags.DEFINE_string("kitti_dir",                  None, "ath to the KITTI dataset directory")
flags.DEFINE_string("init_ckpt_file",             None, "Specific checkpoint file to initialize from")
flags.DEFINE_integer("batch_size",                   4, "The size of of a sample batch")
flags.DEFINE_integer("num_threads",                 32, "Number of threads for data loading")
flags.DEFINE_integer("img_height",                 128, "Image height")
flags.DEFINE_integer("img_width",                  416, "Image width")
flags.DEFINE_integer("seq_length",                   3, "Sequence length for each example")
flags.DEFINE_float("min_depth",                   1e-3, "Threshold for minimum depth")
flags.DEFINE_float("max_depth",                     80, "Threshold for maximum depth")

##### Training Configurations #####
flags.DEFINE_string("checkpoint_dir",               "", "Directory name to save the checkpoints")
flags.DEFINE_float("learning_rate",             0.0002, "Learning rate for adam")
flags.DEFINE_integer("max_to_keep",                 20, "Maximum number of checkpoints to save")
flags.DEFINE_integer("max_steps",               300000, "Maximum number of training iterations")
flags.DEFINE_integer("save_ckpt_freq",            5000, "Save the checkpoint model every save_ckpt_freq iterations")
flags.DEFINE_float("alpha_recon_image",           0.85, "Alpha weight between SSIM and L1 in reconstruction loss, loss rw")

##### Configurations about DepthNet & PoseNet of GeoNet #####
flags.DEFINE_string("dispnet_encoder",      "resnet50", "Type of encoder for dispnet, vgg or resnet50")
flags.DEFINE_boolean("scale_normalize",          False, "Spatially normalize depth prediction")
flags.DEFINE_float("rigid_warp_weight",            1.0, "Weight for warping by rigid flow")
flags.DEFINE_float("disp_smooth_weight",           0.5, "Weight for disp smoothness, lambda ds")

##### Configurations about ResFlowNet of GeoNet (or DirFlowNetS) #####
flags.DEFINE_string("flownet_type",         "residual", "type of flownet, residual or direct")
flags.DEFINE_float("flow_warp_weight",             1.0, "Weight for warping by full flow")
flags.DEFINE_float("flow_smooth_weight",           0.2, "Weight for flow smoothness, lambda fs")
flags.DEFINE_float("flow_consistency_weight",      0.2, "Weight for bidirectional flow consistency, lambda gc")
flags.DEFINE_float("flow_consistency_alpha",       3.0, "Alpha for flow consistency check, lambda gc")
flags.DEFINE_float("flow_consistency_beta",       0.05, "Beta for flow consistency check, lambda gc")

##### Testing Configurations #####
flags.DEFINE_string("output_dir",                 None, "Test result output directory")
flags.DEFINE_string("depth_test_split",        "eigen", "KITTI depth split, eigen or stereo")
flags.DEFINE_integer("pose_test_seq",                9, "KITTI Odometry Sequence ID to test")

# Additional parameters
flags.DEFINE_integer("num_source",     None,    "add configuration")
flags.DEFINE_integer("num_scales",     None,    "add configuration")
flags.DEFINE_boolean("add_flownet",    None,    "add configuration")
flags.DEFINE_boolean("add_dispnet",    None,    "add configuration")
flags.DEFINE_boolean("add_posenet",    None,    "add configuration")

opt = flags.FLAGS

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def convert_disps_to_depths_stereo(gt_disparities, pred_depths):
    gt_depths = []
    pred_depths_resized = []
    pred_disparities_resized = []
    
    for i in range(len(gt_disparities)):
        gt_disp = gt_disparities[i]
        height, width = gt_disp.shape

        pred_depth = pred_depths[i]
        pred_depth = cv2.resize(pred_depth, (width, height), interpolation=cv2.INTER_LINEAR)

        pred_disparities_resized.append(1./pred_depth) 

        mask = gt_disp > 0

        gt_depth = width_to_focal[width] * 0.54 / (gt_disp + (1.0 - mask))
        #pred_depth = width_to_focal[width] * 0.54 / pred_disp

        gt_depths.append(gt_depth)
        pred_depths_resized.append(pred_depth)
    
    return gt_depths, pred_depths_resized, pred_disparities_resized

def test_depth_all(opt):
    # NOTE: only tested with eigen split

    ##### load testing list #####
    with open('data/kitti/test_files_%s.txt' % opt.depth_test_split, 'r') as f:
        test_files = f.readlines()
        test_files = [opt.dataset_dir + t[:-1] for t in test_files]

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    # split the whole file into subsets

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

                pred = sess.run(fetches, feed_dict={input_uint8_tgt: inputs_tgt, input_uint8_src: inputs_src})
            
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

    return pred_all


def locate_cktp(opt, ckpt_dir):
    cktp_dict = {}
    # More decent way is glob syntax matching
    ckpt_list = list(set([ckpt.split(".")[0] for ckpt in os.listdir(ckpt_dir) if ckpt.split(".")[0].startswith("model")]))
    ckpt_list = [int(ckpt.split("-")[1]) for ckpt in ckpt_list]
    ckpt_list.sort()

    for ckpt in ckpt_list:
        ckpt_path = os.path.join(ckpt_dir, "model-"+str(ckpt))
        cktp_dict[ckpt] = ckpt_path

    return cktp_dict

def eval_depth_metric(opt, pred_depths, f):
    test_file_list = 'data/kitti/test_files_%s.txt' % opt.depth_test_split
    if opt.depth_test_split == 'eigen':
        test_files = read_text_lines(test_file_list)
        gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, opt.kitti_dir)
        num_test = len(im_files)
        gt_depths = []
        pred_depths_resized = []
        for t_id in range(num_test):
            camera_id = cams[t_id]  # 2 is left, 3 is right
            pred_depths_resized.append(
                cv2.resize(pred_depths[t_id], 
                           (im_sizes[t_id][1], im_sizes[t_id][0]), 
                           interpolation=cv2.INTER_LINEAR))

            depth = generate_depth_map(gt_calib[t_id], 
                                       gt_files[t_id], 
                                       im_sizes[t_id], 
                                       camera_id, 
                                       False, 
                                       True)
            gt_depths.append(depth.astype(np.float32))
        pred_depths = pred_depths_resized
    else:
        num_test = 200
        gt_disparities = load_gt_disp_kitti(opt.kitti_dir)
        gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_stereo(gt_disparities, pred_depths)
    
    rms     = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel  = np.zeros(num_test, np.float32)
    d1_all  = np.zeros(num_test, np.float32)
    a1      = np.zeros(num_test, np.float32)
    a2      = np.zeros(num_test, np.float32)
    a3      = np.zeros(num_test, np.float32)

    for i in range(num_test):    
        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])

        if opt.depth_test_split == 'eigen':

            mask = np.logical_and(gt_depth > opt.min_depth, 
                                  gt_depth < opt.max_depth)

            # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
            # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
            gt_height, gt_width = gt_depth.shape
            crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                             0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        if opt.depth_test_split == 'stereo':
            gt_disp = gt_disparities[i]
            mask = gt_disp > 0
            pred_disp = pred_disparities_resized[i]

            disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
            bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
            d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

        # Scale matching
        scalor = np.median(gt_depth[mask])/np.median(pred_depth[mask])
        pred_depth[mask] *= scalor

        pred_depth[pred_depth < opt.min_depth] = opt.min_depth
        pred_depth[pred_depth > opt.max_depth] = opt.max_depth
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
            compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    f.write("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))
    f.write("\n")
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))    

def main(_):
    opt.num_source = opt.seq_length - 1  
    # depth: 3-1=2 or odometry: 5-1=4

    opt.num_scales = 4

    opt.add_flownet = opt.mode in ['train_flow', 'test_flow']
    opt.add_dispnet = opt.add_flownet and opt.flownet_type == 'residual' \
        or opt.mode in ['train_rigid', 'test_depth', 'test_depth_delta']
    opt.add_posenet = opt.add_flownet and opt.flownet_type == 'residual' \
        or opt.mode in ['train_rigid', 'test_pose']
    
    make_dir(opt.output_dir)

    cktp_dict = locate_cktp(opt, opt.ckpt_dir)
    f = open(os.path.join(opt.output_dir, "metric_results.txt"), "a")
    f.write("--------------------------------------------------------------------------------------------------")
    f.write("\n")
    f.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    f.write("\n")
    f.close()

    for ckpt, ckpt_path in cktp_dict.items():
        print(ckpt_path)
        f = open(os.path.join(opt.output_dir, "metric_results.txt"), "a")
        f.write("--------------------------------------------------------------------------------------------------")
        f.write("\n")
        f.write(ckpt_path)
        f.write("\n")
        opt.init_ckpt_file = ckpt_path
        tf.reset_default_graph()
        model_output_npy = test_depth_all(opt)
        eval_depth_metric(opt, model_output_npy, f)
        f.close()

if __name__ == "__main__":
    tf.compat.v1.app.run()