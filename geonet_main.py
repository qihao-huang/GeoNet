# -*- coding: utf-8 -*-
from __future__ import division
import os

import time
import random
import pprint
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from geonet_model import *
from geonet_test_depth import *
from geonet_test_pose import *
from geonet_test_flow import *
from data_loader import DataLoader

flags = tf.app.flags
flags.DEFINE_boolean("delta_mode",                False, "whether taining the delta xyz")
flags.DEFINE_string("mode",                         "", "(train_rigid, train_flow) or (test_depth, test_pose, test_flow)")
flags.DEFINE_string("dataset_dir",                  "", "Dataset directory")
flags.DEFINE_string("init_ckpt_file",             None, "Specific checkpoint file to initialize from")
flags.DEFINE_integer("batch_size",                   4, "The size of of a sample batch")
flags.DEFINE_integer("num_threads",                 32, "Number of threads for data loading")
flags.DEFINE_integer("img_height",                 128, "Image height")
flags.DEFINE_integer("img_width",                  416, "Image width")
flags.DEFINE_integer("seq_length",                   3, "Sequence length for each example")
flags.DEFINE_string("log_savedir",                  "", "log directory in save graph and loss")

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

def train():
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    if not os.path.exists(opt.log_savedir):
        os.makedirs(opt.log_savedir)

    with tf.Graph().as_default():
        # Data Loader
        print("# Data Loader")
        loader = DataLoader(opt)

        # tgt_image:       (4,128,416,3)
        # src_image_stack: (4,128,416,6)
        # intrinsics:      (4,4,3,3)
        tgt_image, src_image_stack, intrinsics = loader.load_train_val_batch("train")

        # Build Model
        print("# Build Model")
        model = GeoNetModel(opt, tgt_image, src_image_stack, intrinsics)
        loss = model.total_loss

        # TODO: validation set loss
        # val_tgt_image, val_src_image_stack, val_intrinsics = loader.load_train_val_batch("val")
        # val_model = GeoNetModel(opt, val_tgt_image, val_src_image_stack, val_intrinsics)
        # val_loss = val_model.total_loss

        # Train Op
        print('# Train Op')
        if opt.mode == 'train_flow' and opt.flownet_type == "residual":
            # we pretrain DepthNet & PoseNet, then finetune ResFlowNetS
            train_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, "flow_net")
            vars_to_restore = slim.get_variables_to_restore(
                include=["depth_net", "pose_net"])
        else:
            train_vars = [var for var in tf.trainable_variables()]
            vars_to_restore = slim.get_model_variables()

        # If train from checkpoint file
        if opt.init_ckpt_file != None:
            init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
                opt.init_ckpt_file, vars_to_restore)

        print("# Optimizer")
        optim = tf.train.AdamOptimizer(opt.learning_rate, 0.9)
        train_op = slim.learning.create_train_op(loss, optim,
                                                 variables_to_train=train_vars)

        # Global Step
        print("# Global Step")
        global_step = tf.Variable(0, name='global_step', trainable=False)
        incr_global_step = tf.assign(global_step, global_step+1)

        # Parameter Count
        print("# Parameter Count")
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v))
                                         for v in train_vars])

        # Saver
        print("# Saver")
        saver = tf.train.Saver([var for var in tf.model_variables()] +
                               [global_step],
                               max_to_keep=opt.max_to_keep)

        # Session
        print("# Session")
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with sv.managed_session(config=config) as sess:
            print('Trainable variables: ')
            for var in train_vars:
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))

            if opt.init_ckpt_file != None:
                sess.run(init_assign_op, init_feed_dict)
            start_time = time.time()

            print("------------------------")
            print("entering into iterations")

            for step in range(1, opt.max_steps):
                fetches = {
                        "train": train_op,
                        "global_step": global_step,
                        "incr_global_step": incr_global_step,
                    }
            
                fetches["depth"] = model.pred_depth[0] # fetch depth
                fetches["pose"] = model.pred_poses # fetch pose
                
                if opt.delta_mode:
                    fetches["delta_xyz"] = model.delta_xyz[0] # fetch delta

                if step % 100 == 0:
                    fetches["loss"] = loss

                results = sess.run(fetches)

                save_tmp_name = str(time.time()) # '1585880463.4654446'
                np.save(os.path.join(opt.log_savedir, "depth", save_tmp_name), results["depth"])
                np.save(os.path.join(opt.log_savedir, "pose", save_tmp_name), results["pose"])

                if opt.delta_mode:
                    np.save(os.path.join(opt.log_savedir, "delta", save_tmp_name), results["delta_xyz"])
                
                if step % 100 == 0:
                    time_per_iter = (time.time() - start_time) / 100
                    start_time = time.time()
                    print('Iteration: [%7d] | Time: %4.4fs/iter | Loss: %.3f'
                          % (step, time_per_iter, results["loss"]))

                if step % opt.save_ckpt_freq == 0:
                    saver.save(sess, os.path.join(
                        opt.checkpoint_dir, 'model'), global_step=step)
                    
                    # TODO: do validation here
                    # eval_time = time.time()

                    # time_eval = (time.time() - eval_time)
                    # print('Evaluation: [%7d] | Time: %4.4fs | Loss: %.3f'
                    #       % (step, time_eval, val_results["loss"]))


def main(_):
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # from tensorflow.python.client import device_lib
    # print(device_lib.list_local_devices())

    # tf.test.is_gpu_available()
    # tf.test.gpu_device_name()

    opt.num_source = opt.seq_length - 1  
    # depth: 3-1=2 or 
    # odmetery: 5-1=4
    opt.num_scales = 4

    opt.add_flownet = opt.mode in ['train_flow', 'test_flow']
    opt.add_dispnet = opt.add_flownet and opt.flownet_type == 'residual' \
        or opt.mode in ['train_rigid', 'test_depth']
    opt.add_posenet = opt.add_flownet and opt.flownet_type == 'residual' \
        or opt.mode in ['train_rigid', 'test_pose']

    if opt.mode in ['train_rigid', 'train_flow']:
        train()
    elif opt.mode == 'test_depth':
        test_depth(opt)
    elif opt.mode == 'test_pose':
        test_pose(opt)
    elif opt.mode == 'test_flow':
        test_flow(opt)

if __name__ == '__main__':
    tf.app.run()
