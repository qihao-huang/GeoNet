from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from geonet_nets import *
from utils import *

class GeoNetModel(object):
    def __init__(self, opt, tgt_image, src_image_stack, intrinsics):
        self.opt = opt
        self.tgt_image = self.preprocess_image(tgt_image) # (4, 128, 416, 3)
        self.src_image_stack = self.preprocess_image(src_image_stack) # # (4, 128, 416, 6)
        self.intrinsics = intrinsics # (4, 4, 3, 3)

        self.build_model()

        if not opt.mode in ['train_rigid', 'train_flow']:
            # if test, we don't need build losses function
            return

        self.build_losses()

    # opt.mode: train_rigid, train_flow, test_depth, test_pose, test_flow
    # opt.num_source = opt.seq_length - 1 #  3-1=2 or 5-1=4
    # opt.num_scales = 4, pyramid
    
    # opt.add_flownet = opt.mode in ['train_flow', 'test_flow']
    
    # dispnet: if train/test depth or if train residual flow
    # opt.add_dispnet = opt.add_flownet and opt.flownet_type == 'residual' \
    #                   or opt.mode in ['train_rigid', 'test_depth']

    # posenet: if train/test pose or if train residual flow
    # opt.add_posenet = opt.add_flownet and opt.flownet_type == 'residual' \
    #                   or opt.mode in ['train_rigid', 'test_pose']

    def build_model(self):
        opt = self.opt

        # self.tgt_image: (4, 128, 416, 3)
        # self.tgt_image_pyramid:
        # [(4, 128, 416, 3), (4, 64, 208, 3), (4, 32, 104, 3), (4, 16, 52, 3)]
        self.tgt_image_pyramid = self.scale_pyramid(self.tgt_image, opt.num_scales)

        # tile: duplicate first channel
        # self.tgt_image_tile_pyramid:
        # [opt.num_source, 1, 1, 1] = [2,1,1,1]
        # [(8, 128, 416, 3), (8, 64, 208, 3), (8, 32, 104, 3), (8, 16, 52, 3)]
        self.tgt_image_tile_pyramid = [tf.tile(img, [opt.num_source, 1, 1, 1]) \
                                      for img in self.tgt_image_pyramid]

        # src images concated along batch dimension
        if self.src_image_stack != None:
            # (8, 128, 416, 3), src_1 and src_2 concat in axis=0
            self.src_image_concat = tf.concat([self.src_image_stack[:,:,:,3*i:3*(i+1)] \
                                    for i in range(opt.num_source)], axis=0)
            
            # [(8, 128, 416, 3), (8, 64, 208, 3), (8, 32, 104, 3), (8, 16, 52, 3)]
            self.src_image_concat_pyramid = self.scale_pyramid(self.src_image_concat, opt.num_scales)

        if opt.add_dispnet:
            self.build_dispnet()

        if opt.add_posenet:
            self.build_posenet()

        if opt.add_dispnet and opt.add_posenet:
            self.build_rigid_flow_warping()

        if opt.add_flownet:
            self.build_flownet()
            if opt.mode == 'train_flow':
                self.build_full_flow_warping()
                # Weight for bidirectional flow consistency
                # flags.DEFINE_float("flow_consistency_weight", 0.2, "Weight for bidirectional flow consistency") # lambda gc
                if opt.flow_consistency_weight > 0:
                    self.build_flow_consistency()

    # ----------------------------------------------------------------------------------------------
    def build_dispnet(self):
        opt = self.opt

        # build dispnet_inputs
        if opt.mode == 'test_depth':
            # for test_depth mode we only predict the depth of the target image
            self.dispnet_inputs = self.tgt_image
        else:
            # multiple depth predictions; 
            # tgt: disp[:bs,:,:,:] src.i: disp[bs*(i+1):bs*(i+2),:,:,:]
            # >>> a = [1,2,3,4,5,6,7,8,9,10,11,12]
            # >>> a[0:4]            # [1, 2, 3, 4]
            # >>> a[4:8]            # [5, 6, 7, 8]
            # >>> a[8:12]           # [9, 10, 11, 12]
            self.dispnet_inputs = self.tgt_image
            for i in range(opt.num_source):
                self.dispnet_inputs = tf.concat([self.dispnet_inputs, self.src_image_stack[:,:,:,3*i:3*(i+1)]], axis=0)
        
        # build dispnet
        # self.dispnet_inputs: (12 (tgt, src_1, src_2), 128, 416, 3), axis0: 0:4 tgt, 4:8 src_1, 8:12 src_2
        # self.pred_disp:
        # [(12, 128, 416, 1) , (12, 64, 208, 1) , (12, 32, 104, 1) , (12, 16, 52, 1) ]
        # self.delta_xyz:
        # [(4, 128, 416, 12), (4, 64, 208, 12) , (4, 32, 104, 12) , (4, 16, 52, 12) ]
        self.pred_disp, self.delta_xyz = disp_net(opt, self.dispnet_inputs)

        if opt.scale_normalize:
            # As proposed in https://arxiv.org/abs/1712.00175, this can 
            # bring improvement in depth estimation, but not included in our paper.
            self.pred_disp = [self.spatial_normalize(disp) for disp in self.pred_disp]

        self.pred_depth = [1./d for d in self.pred_disp]

    def build_posenet(self):
        opt = self.opt

        # build posenet_inputs, (4,128,416,9(tgt,src_1,src_2))
        self.posenet_inputs = tf.concat([self.tgt_image, self.src_image_stack], axis=3)
        
        # build posenet
        # self.pred_poses: (4, 2(tgt->src_1, tgt->src_2), 6(tx,ty,tx,rx,ry,rz))
        self.pred_poses = pose_net(opt, self.posenet_inputs)

    def build_rigid_flow_warping(self):
        opt = self.opt
        bs = opt.batch_size # 4

        # build rigid flow (fwd: tgt->src, bwd: src->tgt)
        self.fwd_rigid_flow_pyramid = []
        self.bwd_rigid_flow_pyramid = []
        for s in range(opt.num_scales):
            # TODO: 
            # for deltax_xyz:
            # i=0: 0:3 -> fwd, 6:9 -> bwd
            # i=1: 3:6 -> fwd, 9:12 -> bwd
            for i in range(opt.num_source):
                
                # self.pred_depth:
                # [(12, 128, 416, 1) , (12, 64, 208, 1) , (12, 32, 104, 1) , (12, 16, 52, 1) ]
                # self.pred_depth[s(0:3)][:bs], 0:4: the whole batch of tgt
                # tf.squeeze(): (4, 128, 416, 1) -> (4, 128, 416) 
                # self.pred_poses[:,0,:]: tgt->src_1
                # self.pred_poses[:,1,:]: tgt->src_2
                # fwd_rigid_flow shape: (4, 128, 416, 2)
                fwd_rigid_flow = compute_rigid_flow(tf.squeeze(self.pred_depth[s][:bs], axis=3),
                                 self.delta_xyz[s][:,:,:,3*(i):3*(i+1)], self.pred_poses[:,i,:], 
                                 self.intrinsics[:,s,:,:], False)
                
                # backward: src_1 -> tgt, src_2 -> tgt
                # src_1: 4:8, src_2: 8:12
                # bwd_rigid_flow shape: (4, 128, 416, 2)
                bwd_rigid_flow = compute_rigid_flow(tf.squeeze(self.pred_depth[s][bs*(i+1):bs*(i+2)], axis=3),
                                 self.delta_xyz[s][:,:,:,3*(i+2):3*(i+3)], self.pred_poses[:,i,:], 
                                 self.intrinsics[:,s,:,:], True)
                if not i:
                    fwd_rigid_flow_concat = fwd_rigid_flow
                    bwd_rigid_flow_concat = bwd_rigid_flow
                else:
                    fwd_rigid_flow_concat = tf.concat([fwd_rigid_flow_concat, fwd_rigid_flow], axis=0)
                    bwd_rigid_flow_concat = tf.concat([bwd_rigid_flow_concat, bwd_rigid_flow], axis=0)

            # tgt -> src_1 concat tgt -> src_2 in axis = 0
            # fwd_rigid_flow_concat shape: (8, 128, 416, 2)
            self.fwd_rigid_flow_pyramid.append(fwd_rigid_flow_concat)
            self.bwd_rigid_flow_pyramid.append(bwd_rigid_flow_concat)
        
        # len(self.fwd_rigid_flow_pyramid) = 4 
        # self.fwd_rigid_flow_pyramid[0]: (8, 128, 416, 2)
        # warping by rigid flow
        self.fwd_rigid_warp_pyramid = [flow_warp(self.src_image_concat_pyramid[s], self.fwd_rigid_flow_pyramid[s]) \
                                      for s in range(opt.num_scales)]
        self.bwd_rigid_warp_pyramid = [flow_warp(self.tgt_image_tile_pyramid[s], self.bwd_rigid_flow_pyramid[s]) \
                                      for s in range(opt.num_scales)]

        # compute reconstruction error  
        self.fwd_rigid_error_pyramid = [self.image_similarity(self.fwd_rigid_warp_pyramid[s], self.tgt_image_tile_pyramid[s]) \
                                       for s in range(opt.num_scales)]      
        self.bwd_rigid_error_pyramid = [self.image_similarity(self.bwd_rigid_warp_pyramid[s], self.src_image_concat_pyramid[s]) \
                                       for s in range(opt.num_scales)]
    
    # ----------------------------------------------------------------------------------------------
    def build_flownet(self):
        opt = self.opt

        # build flownet_inputs
        self.fwd_flownet_inputs = tf.concat([self.tgt_image_tile_pyramid[0], self.src_image_concat_pyramid[0]], axis=3)
        self.bwd_flownet_inputs = tf.concat([self.src_image_concat_pyramid[0], self.tgt_image_tile_pyramid[0]], axis=3)

        # TODO: direct or residual ?
        if opt.flownet_type == 'residual':
            self.fwd_flownet_inputs = tf.concat([self.fwd_flownet_inputs, 
                                      self.fwd_rigid_warp_pyramid[0], # warped RGB
                                      self.fwd_rigid_flow_pyramid[0], # rigid flow
                                      self.L2_norm(self.fwd_rigid_error_pyramid[0])], axis=3) 
            self.bwd_flownet_inputs = tf.concat([self.bwd_flownet_inputs,
                                      self.bwd_rigid_warp_pyramid[0],
                                      self.bwd_rigid_flow_pyramid[0],
                                      self.L2_norm(self.bwd_rigid_error_pyramid[0])], axis=3)
        self.flownet_inputs = tf.concat([self.fwd_flownet_inputs, self.bwd_flownet_inputs], axis=0)
        
        # build flownet
        self.pred_flow = flow_net(opt, self.flownet_inputs)

        # unnormalize pyramid flow back into pixel metric
        for s in range(opt.num_scales):
            curr_bs, curr_h, curr_w, _ = self.pred_flow[s].get_shape().as_list()
            scale_factor = tf.cast(tf.constant([curr_w, curr_h], shape=[1,1,1,2]), 'float32')
            scale_factor = tf.tile(scale_factor, [curr_bs, curr_h, curr_w, 1])
            self.pred_flow[s] = self.pred_flow[s] * scale_factor

        # split forward/backward flows
        self.fwd_full_flow_pyramid = [self.pred_flow[s][:opt.batch_size*opt.num_source] for s in range(opt.num_scales)]
        self.bwd_full_flow_pyramid = [self.pred_flow[s][opt.batch_size*opt.num_source:] for s in range(opt.num_scales)]

        # residual flow postprocessing
        if opt.flownet_type == 'residual':
            self.fwd_full_flow_pyramid = [self.fwd_full_flow_pyramid[s] + self.fwd_rigid_flow_pyramid[s] for s in range(opt.num_scales)]
            self.bwd_full_flow_pyramid = [self.bwd_full_flow_pyramid[s] + self.bwd_rigid_flow_pyramid[s] for s in range(opt.num_scales)]   

    def build_full_flow_warping(self):
        opt = self.opt
        
        # warping by full flow
        self.fwd_full_warp_pyramid = [flow_warp(self.src_image_concat_pyramid[s], self.fwd_full_flow_pyramid[s]) \
                                      for s in range(opt.num_scales)]
        self.bwd_full_warp_pyramid = [flow_warp(self.tgt_image_tile_pyramid[s], self.bwd_full_flow_pyramid[s]) \
                                      for s in range(opt.num_scales)]

        # compute reconstruction error  
        self.fwd_full_error_pyramid = [self.image_similarity(self.fwd_full_warp_pyramid[s], self.tgt_image_tile_pyramid[s]) \
                                       for s in range(opt.num_scales)]      
        self.bwd_full_error_pyramid = [self.image_similarity(self.bwd_full_warp_pyramid[s], self.src_image_concat_pyramid[s]) \
                                       for s in range(opt.num_scales)]    

    def build_flow_consistency(self):
        opt = self.opt

        # warp pyramid full flow
        self.bwd2fwd_flow_pyramid = [flow_warp(self.bwd_full_flow_pyramid[s], self.fwd_full_flow_pyramid[s]) \
                                    for s in range(opt.num_scales)]
        self.fwd2bwd_flow_pyramid = [flow_warp(self.fwd_full_flow_pyramid[s], self.bwd_full_flow_pyramid[s]) \
                                    for s in range(opt.num_scales)]

        # calculate flow consistency
        self.fwd_flow_diff_pyramid = [tf.abs(self.bwd2fwd_flow_pyramid[s] + self.fwd_full_flow_pyramid[s]) for s in range(opt.num_scales)]
        self.bwd_flow_diff_pyramid = [tf.abs(self.fwd2bwd_flow_pyramid[s] + self.bwd_full_flow_pyramid[s]) for s in range(opt.num_scales)]

        # build flow consistency condition
        self.fwd_consist_bound = [opt.flow_consistency_beta * self.L2_norm(self.fwd_full_flow_pyramid[s]) * 2**s for s in range(opt.num_scales)]
        self.bwd_consist_bound = [opt.flow_consistency_beta * self.L2_norm(self.bwd_full_flow_pyramid[s]) * 2**s for s in range(opt.num_scales)]
        self.fwd_consist_bound = [tf.stop_gradient(tf.maximum(v, opt.flow_consistency_alpha)) for v in self.fwd_consist_bound]
        self.bwd_consist_bound = [tf.stop_gradient(tf.maximum(v, opt.flow_consistency_alpha)) for v in self.bwd_consist_bound]

        # build flow consistency mask
        self.noc_masks_src = [tf.cast(tf.less(self.L2_norm(self.bwd_flow_diff_pyramid[s]) * 2**s, 
                             self.bwd_consist_bound[s]), tf.float32) for s in range(opt.num_scales)]
        self.noc_masks_tgt = [tf.cast(tf.less(self.L2_norm(self.fwd_flow_diff_pyramid[s]) * 2**s,
                             self.fwd_consist_bound[s]), tf.float32) for s in range(opt.num_scales)]

    # ----------------------------------------------------------------------------------------------
    def build_losses(self):
        opt = self.opt
        bs = opt.batch_size
        
        rigid_warp_loss = 0
        disp_smooth_loss = 0
        flow_warp_loss = 0
        flow_smooth_loss = 0
        flow_consistency_loss = 0

        # opt.num_scales = 4
        for s in range(opt.num_scales):
            # rigid_warp_loss
            # flags.DEFINE_float("rigid_warp_weight",  1.0, "Weight for warping by rigid flow") 
            if opt.mode == 'train_rigid' and opt.rigid_warp_weight > 0:
                # TODO: why opt.num_source/2
                rigid_warp_loss += opt.rigid_warp_weight*opt.num_source/2 * \
                                (tf.reduce_mean(self.fwd_rigid_error_pyramid[s]) + \
                                 tf.reduce_mean(self.bwd_rigid_error_pyramid[s]))

            # disp_smooth_loss: edge-aware depth smoothness loss
            # flags.DEFINE_float("disp_smooth_weight", 0.5, "Weight for disp smoothness") # lambda ds
            if opt.mode == 'train_rigid' and opt.disp_smooth_weight > 0:
                # TODO: why /(2**s)
                # TODO: why tf.concat
                disp_smooth_loss += opt.disp_smooth_weight/(2**s) * self.compute_smooth_loss(self.pred_disp[s],
                                tf.concat([self.tgt_image_pyramid[s], self.src_image_concat_pyramid[s]], axis=0))

            # flow_warp_loss
            # flags.DEFINE_float("flow_warp_weight", 1.0, "Weight for warping by full flow") 
            # flags.DEFINE_float("flow_consistency_weight", 0.2, "Weight for bidirectional flow consistency") # lambda gc
            if opt.mode == 'train_flow' and opt.flow_warp_weight > 0:
                if opt.flow_consistency_weight == 0:
                    flow_warp_loss += opt.flow_warp_weight*opt.num_source/2 * \
                                (tf.reduce_mean(self.fwd_full_error_pyramid[s]) + \
                                 tf.reduce_mean(self.bwd_full_error_pyramid[s]))
                else:
                    # TODO: fwd_full_error_pyramid
                    flow_warp_loss += opt.flow_warp_weight*opt.num_source/2 * \
                                (tf.reduce_sum(tf.reduce_mean(self.fwd_full_error_pyramid[s], axis=3, keep_dims=True) * \
                                 self.noc_masks_tgt[s]) / tf.reduce_sum(self.noc_masks_tgt[s]) + \
                                 tf.reduce_sum(tf.reduce_mean(self.bwd_full_error_pyramid[s], axis=3, keep_dims=True) * \
                                 self.noc_masks_src[s]) / tf.reduce_sum(self.noc_masks_src[s]))

            # flow_smooth_loss
            # flags.DEFINE_float("flow_smooth_weight", 0.2, "Weight for flow smoothness") # lambda_fs
            if opt.mode == 'train_flow' and opt.flow_smooth_weight > 0:
                # TODO: why s+1
                # TODO: fwd <-> tgt, bwd <-> src
                flow_smooth_loss += opt.flow_smooth_weight/(2**(s+1)) * \
                                (self.compute_flow_smooth_loss(self.fwd_full_flow_pyramid[s], self.tgt_image_tile_pyramid[s]) +
                                self.compute_flow_smooth_loss(self.bwd_full_flow_pyramid[s], self.src_image_concat_pyramid[s]))

            # flow_consistency_loss
            if opt.mode == 'train_flow' and opt.flow_consistency_weight > 0:
                # TODO: fwd_flow_diff_pyramid
                flow_consistency_loss += opt.flow_consistency_weight/2 * \
                                (tf.reduce_sum(tf.reduce_mean(self.fwd_flow_diff_pyramid[s] , axis=3, keep_dims=True) * \
                                 self.noc_masks_tgt[s]) / tf.reduce_sum(self.noc_masks_tgt[s]) + \
                                 tf.reduce_sum(tf.reduce_mean(self.bwd_flow_diff_pyramid[s] , axis=3, keep_dims=True) * \
                                 self.noc_masks_src[s]) / tf.reduce_sum(self.noc_masks_src[s]))

        # TODO: regularization_loss ?
        regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
        self.total_loss = 0  # regularization_loss
        if opt.mode == 'train_rigid':
            # rw + ds
            self.total_loss += rigid_warp_loss + disp_smooth_loss
        if opt.mode == 'train_flow':
            # rw + fs + gc
            self.total_loss += flow_warp_loss + flow_smooth_loss + flow_consistency_loss

    # ----------------------------------------------------------------------------------------------

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def image_similarity(self, x, y):
        return self.opt.alpha_recon_image * self.SSIM(x, y) + (1-self.opt.alpha_recon_image) * tf.abs(x-y)

    def L2_norm(self, x, axis=3, keep_dims=True):
        curr_offset = 1e-10
        l2_norm = tf.norm(tf.abs(x) + curr_offset, axis=axis, keep_dims=keep_dims)
        return l2_norm

    # if opt.scale_normalize is adopted
    def spatial_normalize(self, disp):
        _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
        disp_mean = tf.reduce_mean(disp, axis=[1,2,3], keep_dims=True)
        disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
        return disp/disp_mean

    def scale_pyramid(self, img, num_scales):
        if img == None:
            return None
        else:
            scaled_imgs = [img]
            _, h, w, _ = img.get_shape().as_list()
            # i: 0, 1, 2
            for i in range(num_scales - 1):
                ratio = 2 ** (i + 1) # 2, 4, 8
                nh = int(h / ratio)
                nw = int(w / ratio)
                # tmp = tf.image.resize_area(img, [nh, nw])
                # print("tmp shape: ", tmp.get_shape().as_list())
                # scaled_imgs.append(tmp)
                scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
            
            return scaled_imgs

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    # edge-aware smootheness loss
    def compute_smooth_loss(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))

    def compute_flow_smooth_loss(self, flow, img):
        smoothness = 0
        for i in range(2):
            smoothness += self.compute_smooth_loss(tf.expand_dims(flow[:,:,:,i], -1), img)
        return smoothness/2

    def preprocess_image(self, image):
        # Assuming input image is uint8
        if image == None:
            return None
        else:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # TODO: why? *2-1 in preprocess
            return image * 2. -1.

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)
