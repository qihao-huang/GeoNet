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
from geonet_test_depth_detla import *
from geonet_test_pose import *
from geonet_test_flow import *
from data_loader import DataLoader

flags = tf.app.flags
####
flags.DEFINE_boolean("save_intermedia",           False, "whether to save the intermediate vars")
flags.DEFINE_boolean("delta_mode",                False, "whether to train the delta xyz")
####
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

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def train():
    seed = 8964
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("\x1b[6;30;42m" + "# flags" + "\x1b[0m")
    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    make_dir(opt.checkpoint_dir)
    make_dir(opt.log_savedir)

    with tf.Graph().as_default():
        # Data Loader
        print("\x1b[6;30;42m" + "# Data Loader" + "\x1b[0m")
        loader = DataLoader(opt)

        # tgt_image:       (4,128,416,3)
        # src_image_stack: (4,128,416,6)
        # intrinsics:      (4,4,3,3)
        tgt_image, src_image_stack, intrinsics = loader.load_train_val_batch("train")

        # Build Model
        print("\x1b[6;30;42m" + "# Build Model" + "\x1b[0m")
        model = GeoNetModel(opt, tgt_image, src_image_stack, intrinsics)
        loss = model.total_loss
        tf.compat.v1.summary.scalar("loss", loss)
        merged_summary_op = tf.compat.v1.summary.merge_all()

        # skip those self-defined parameters
        skip_para = ['depth_net/depth_net/delta_mod/conv_1//weights:0', 'depth_net/depth_net/delta_mod/conv_1//biases:0',
                     'depth_net/depth_net/delta_mod/conv_2//weights:0', 'depth_net/depth_net/delta_mod/conv_2//biases:0',
                     'depth_net/depth_net/delta_mod/conv_3//weights:0', 'depth_net/depth_net/delta_mod/conv_3//biases:0',
                     'depth_net/depth_net/delta_mod/conv_4//weights:0', 'depth_net/depth_net/delta_mod/conv_4//biases:0']

        # Train Op
        print("\x1b[6;30;42m" + "# Train Op" + "\x1b[0m")
        if opt.mode == 'train_flow' and opt.flownet_type == "residual":
            # we pretrain DepthNet & PoseNet, then finetune ResFlowNetS
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "flow_net")
            vars_to_restore = slim.get_variables_to_restore(include=["depth_net", "pose_net"])
        else:
            train_vars = [var for var in tf.compat.v1.trainable_variables()]
            # vars_to_restore = slim.get_model_variables()
            # Partially Restoring Models：
            vars_to_restore = slim.get_variables_to_restore(exclude=skip_para)

        # vars_to_restore
        print("\x1b[6;30;42m" + "# vars_to_restore" + "\x1b[0m")
        for v in vars_to_restore:
            print(v)

        print("----------------")

        print("\x1b[6;30;42m" + "# train_vars" + "\x1b[0m")
        for v in train_vars:
            print(v)

        # If train from checkpoint file
        if opt.init_ckpt_file != None:
            init_assign_op, init_feed_dict = slim.assign_from_checkpoint(opt.init_ckpt_file, vars_to_restore)

        print("\x1b[6;30;42m" + "# Optimizer" + "\x1b[0m")
        optim = tf.compat.v1.train.AdamOptimizer(opt.learning_rate, 0.9)
        train_op = slim.learning.create_train_op(loss, optim, variables_to_train=train_vars)

        # Global Step
        print("\x1b[6;30;42m" + "# Global Step" + "\x1b[0m")
        global_step = tf.Variable(0, name='global_step', trainable=False)
        incr_global_step = tf.compat.v1.assign(global_step, global_step+1)

        # Parameter Count
        print("\x1b[6;30;42m" + "# Parameter Count" + "\x1b[0m")
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in train_vars])

        # Saver
        print("\x1b[6;30;42m" + "# Saver" + "\x1b[0m")
        saver = tf.compat.v1.train.Saver([var for var in tf.compat.v1.model_variables()] +
                               [global_step],
                               max_to_keep=opt.max_to_keep)
                               # 20 files

        # Session
        print("\x1b[6;30;42m" + "# Session" + "\x1b[0m")
        # tf.train.MonitoredTrainingSession
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        with sv.managed_session(config=config) as sess:
            print("\x1b[6;30;42m" + "# Trainable variables" + "\x1b[0m")

            for var in train_vars:
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))

            if opt.init_ckpt_file != None:
                sess.run(init_assign_op, init_feed_dict)

            print("------------------------")
            print("\x1b[6;30;42m" + "# entering into iterations" + "\x1b[0m")
            
            start_time = time.time()

            for step in range(1, opt.max_steps):
                fetches = {
                        "train": train_op,
                        "global_step": global_step,
                        "incr_global_step": incr_global_step,
                    }
            
                fetches["depth"] = model.pred_depth[0] # fetch depth
                fetches["pose"] = model.pred_poses # fetch pose
                fetches["tgt_image"] = model.tgt_image # fetch tgt_image
                fetches["src_image_stack"] = model.src_image_stack # fetch src_image_stack
                
                if opt.delta_mode:
                    fetches["delta_xyz"] = model.delta_xyz[0] # fetch delta

                if step % 100 == 0:
                    fetches["loss"] = loss

                results = sess.run(fetches)
                
                if opt.save_intermedia:
                    save_tmp_name = str(time.time()) # '1585880463.4654446'
                    depth_save_dir = os.path.join(opt.log_savedir, "depth")
                    pose_save_dir = os.path.join(opt.log_savedir, "pose")
                    tgt_image_save_dir = os.path.join(opt.log_savedir, "tgt_image")
                    src_image_stack_save_dir = os.path.join(opt.log_savedir, "src_image_stack")

                    make_dir(depth_save_dir)
                    make_dir(pose_save_dir)
                    make_dir(tgt_image_save_dir)
                    make_dir(src_image_stack_save_dir)
                 
                    np.save(os.path.join(depth_save_dir, save_tmp_name), results["depth"])
                    np.save(os.path.join(pose_save_dir, save_tmp_name), results["pose"])
                    np.save(os.path.join(tgt_image_save_dir, save_tmp_name), results["tgt_image"])
                    np.save(os.path.join(src_image_stack_save_dir, save_tmp_name), results["src_image_stack"])

                    if opt.delta_mode:
                        delta_save_dir = os.path.join(opt.log_savedir, "delta")
                        make_dir(delta_save_dir)
                        np.save(os.path.join(delta_save_dir, save_tmp_name), results["delta_xyz"])
                
                if step % 100 == 0:
                    time_per_iter = (time.time() - start_time) / 100
                    start_time = time.time()
                    print('Iteration: [%7d] | Time: %4.4fs/iter | Loss: %.3f' % (step, time_per_iter, results["loss"]))

                    merged_summary = sess.run(merged_summary_op)
                    sv.summary_computed(sess, merged_summary, global_step=step)

                if step % opt.save_ckpt_freq == 0:
                    saver.save(sess, os.path.join(opt.checkpoint_dir, 'model'), global_step=step)
            
def main(_):
    opt.num_source = opt.seq_length - 1  
    # depth: 3-1=2 or odometry: 5-1=4

    opt.num_scales = 4

    opt.add_flownet = opt.mode in ['train_flow', 'test_flow']
    opt.add_dispnet = opt.add_flownet and opt.flownet_type == 'residual' \
        or opt.mode in ['train_rigid', 'test_depth', 'test_depth_delta']
    opt.add_posenet = opt.add_flownet and opt.flownet_type == 'residual' \
        or opt.mode in ['train_rigid', 'test_pose']

    if opt.mode in ['train_rigid', 'train_flow']:
        train()
    elif opt.mode == 'test_depth':
        test_depth(opt)
    elif opt.mode == 'test_depth_delta':
        test_depth_delta(opt)
    elif opt.mode == 'test_pose':
        test_pose(opt)
    elif opt.mode == 'test_flow':
        test_flow(opt)

if __name__ == '__main__':
    tf.compat.v1.app.run()
