# download the KITTI flow 2015 dataset and its multi-view extension. 
# For replicating our flow results in the paper, a seq_length of 3 is recommended.

# format the testing data
# python kitti_eval/generate_multiview_extension.py \
#     --dataset_dir="/userhome/34/h3567721/dataset/kitti/flow/data_scene_flow_multiview/" \
#     --calib_dir="/userhome/34/h3567721/dataset/kitti/flow/data_scene_flow_calib/" \
#     --dump_root="/userhome/34/h3567721/dataset/kitti/flow/data_scene_flow_mv_dump_test_data/" \
#     --cam_id=02 \
#     --seq_length=3

# test your trained model
python geonet_main.py \
    --mode=test_flow \
    --dataset_dir="/userhome/34/h3567721/dataset/kitti/flow/data_scene_flow_mv_dump_test_data" \
    --init_ckpt_file=/path/to/trained/model/ \
    --flownet_type=direct \
    --batch_size=1 \
    --output_dir="/userhome/34/h3567721/projects/GeoNet/predictions/test_flow"

# evaluation script
python kitti_eval/eval_flow.py \
    --dataset_dir=/path/to/kitti_stereo_2015/ \
    --pred_dir="/userhome/34/h3567721/projects/GeoNet/predictions/eval_flow"