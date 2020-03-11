# depth and pose tasks

# You can switch the network encoder by setting --dispnet_encoder flag, 
# or perform depth scale normalization (see this paper for details) by 
# setting --scale_normalize as True. 
# Note that for replicating depth and pose results, 
# the --seq_length is suggested to be 3 and 5 respectively.

PYTHON="/userhome/34/h3567721/anaconda3/envs/geonet-v/bin/python"

$PYTHON geonet_main.py \
    --mode=train_rigid \
    # --dispnet_encoder \ 
    # --scale_normalize \
    --dataset_dir=/userhome/34/h3567721/dataset/kitti/kitti_raw_eigen/ \
    --checkpoint_dir=/userhome/34/h3567721/project/geonet/checkpoint/ \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=350000 \