# flow task, direct or residual flow learning.
# You can choose to learn direct or residual flow by setting --flownet_type flag. 
# Note that when the --flownet_type is residual, 
# the --init_ckpt_file should be specified to point at a model pretrained on the same dataset with mode of train_rigid. 
# Also a max_steps more than 200 epochs is preferred for learning residual flow.
PYTHON="/userhome/34/h3567721/anaconda3/envs/geonet-v/bin/python"

$PYTHON geonet_main.py \
    --mode=train_flow \
    # --flownet_type \
    --dataset_dir=/path/to/formatted/data/ \
    --checkpoint_dir=/path/to/save/ckpts/ \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --flownet_type=direct \
    --max_steps=400000
