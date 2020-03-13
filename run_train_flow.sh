# flow task, direct or residual flow learning.
# You can choose to learn direct or residual flow by setting --flownet_type flag. 
# When the --flownet_type is residual, 
# the --init_ckpt_file should be specified to point at a model pretrained on the same dataset with mode of train_rigid. 
# Also a max_steps more than 200 epochs is preferred for learning residual flow.

python geonet_main.py \
    --mode=train_flow \
    --dataset_dir="/userhome/34/h3567721/dataset/kitti/kitti_stereo" \
    --checkpoint_dir="/userhome/34/h3567721/projects/GeoNet/checkpoint" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --max_steps=400000 \
    --flownet_type=direct 
    # --init_ckpt_file