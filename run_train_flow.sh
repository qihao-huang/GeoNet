# flow task, direct or residual flow learning.
# You can choose to learn direct or residual flow by setting --flownet_type flag. 
# When the --flownet_type is residual, 
# the --init_ckpt_file should be specified to point at
# a model pretrained on the same dataset with mode of train_rigid. 
# Also a max_steps more than 200 epochs is preferred for learning residual flow.

export CUDA_VISIBLE_DEVICES=0

user_path="/userhome/34/h3567721"

python geonet_main.py \
    --mode=train_flow \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_stereo" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet/checkpoint/checkpoint_flow" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --max_steps=400000 \
    --flownet_type=residual \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet/ccheckpoint/heckpoint_depth/model-240000"