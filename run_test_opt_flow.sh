# download the KITTI flow 2015 dataset and its multi-view extension. 
# For replicating our flow results in the paper, a seq_length of 3 is recommended.

export CUDA_VISIBLE_DEVICES=0

user_path="/userhome/34/h3567721"

# format the testing data
python kitti_eval/generate_multiview_extension.py \
    --dataset_dir=$user_path"/dataset/kitti/flow/data_scene_flow_multiview/" \
    --calib_dir=$user_path"/dataset/kitti/flow/data_scene_flow_calib/" \
    --dump_root=$user_path"/dataset/kitti/flow/data_scene_flow_mv_dump_test_data/" \
    --cam_id=02 \
    --seq_length=3

# if train with --flownet_type=direct, shoule be tested with direct as well
# test your trained model
python geonet_main.py \
    --mode=test_flow \
    --dataset_dir=$user_path"/dataset/kitti/flow/data_scene_flow_mv_dump_test_data/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet/checkpoint_flow/model-395000" \
    --flownet_type=residual \
    --batch_size=1 \
    --output_dir=$user_path"/projects/Depth/GeoNet/predictions/test_flow/"

# evaluation script
python kitti_eval/eval_flow.py \
    --dataset_dir=$user_path"/dataset/kitti/flow/data_scene_flow/" \
    --pred_dir=$user_path"/projects/Depth/GeoNet/predictions/test_flow/model-395000/"

# Mean Noc EPE = 7.6555 
# Mean Noc ACC = 0.6843 
# Mean Occ EPE = 11.7195 
# Mean Occ ACC = 0.6116 