# 6/25 exp3
# 6/25 exp4

import os
from tqdm import tqdm
import detectron2

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow
import torch 

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def format_file_list(data_root, split):
    with open(data_root + '/%s.txt' % split, 'r') as f:
        frames = f.readlines()

    subfolders = [x.split(' ')[0] for x in frames]
    frame_ids = [x.split(' ')[1][:-1] for x in frames]

    image_file_list = [os.path.join(data_root, subfolders[i],
                                    frame_ids[i] + '.jpg') for i in range(len(frames))]
    return image_file_list

thing_classes=['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

def seg_sig_img(img_path):
    im = cv2.imread(img_path)
    seg_img = predictor(im)

    print(img_path)
    
    mask_img = np.zeros((128, 1248, 1), dtype = "uint8")

    print(img_path)
    for i in range(len(seg_img["instances"].pred_masks)):
        img_class = thing_classes[seg_img["instances"].pred_classes[i].cpu()]
        img_score = seg_img["instances"].scores[i].cpu().numpy()
        
        if img_class in thing_classes:
            print("class: ", img_class,  "score: ", img_score)

            mask = (seg_img["instances"].pred_masks[i].cpu() > 0).type(torch.uint8)  
            mask_img += torch.unsqueeze(mask, 2).numpy()*255
            
    return mask_img

def save_seg_img(seg_img, save_dir, save_name):
    cv2.imwrite(os.path.join(save_dir, save_name), seg_img)

if __name__ == "__main__":
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")
    predictor = DefaultPredictor(cfg)

    # pytorch-v 1.4.0
    # DONT use geonet-v
    base_path = "/userhome/34/h3567721/dataset/kitti"
    kitti_raw_eigen_data_dir = os.path.join(base_path, "kitti_raw_eigen")
    image_file_list = format_file_list(kitti_raw_eigen_data_dir, split="train")
    # print(len(image_file_list))
    # 40238 files -> 1248*128 (3 consecutive images 416*128)

    save_base_dir = os.path.join(base_path, "kitti_raw_eigen_seg", "mask")

    make_dir(save_base_dir)

    # load path: ...../kitti/kitti_raw_eigen/2011_09_28_drive_0039_sync_02/0000000285.jpg
    # save path: ...../kitti/kitti_raw_eigen_seg/mask/2011_09_28_drive_0039_sync_02/0000000285.jpg

    # for sig_img_path in image_file_list:
    #     make_dir(os.path.join(save_base_dir, os.path.dirname(sig_img_path).split("/")[-1]))

    for i in range(len(image_file_list)):
        sig_img_path = image_file_list[i]
        seg_img = seg_sig_img(sig_img_path)
        
        save_dir = os.path.join(save_base_dir, os.path.dirname(sig_img_path).split("/")[-1])
        save_name = os.path.basename(sig_img_path)
        save_seg_img(seg_img, save_dir, save_name)