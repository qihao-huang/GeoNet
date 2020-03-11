# download the KITTI flow 2015 dataset and its multi-view extension. 
# For replicating our flow results in the paper, a seq_length of 3 is recommended.

PYTHON="/userhome/34/h3567721/anaconda3/envs/geonet-v/bin/python"

#  format the testing data
$PYTHON kitti_eval/generate_multiview_extension.py \
    --dataset_dir=/path/to/data_scene_flow_multiview/ \
    --calib_dir=/path/to/data_scene_flow_calib/ \
    --dump_root=/path/to/formatted/testdata/ \
    --cam_id=02 \
    --seq_length=3

# test your trained model
$PYTHON geonet_main.py \
    --mode=test_flow \
    --dataset_dir=/path/to/formatted/testdata/ \
    --init_ckpt_file=/path/to/trained/model/ \
    --flownet_type=direct \
    --batch_size=1 \
    --output_dir=/path/to/save/predictions/

# evaluation script
$PYTHON kitti_eval/eval_flow.py \
    --dataset_dir=/path/to/kitti_stereo_2015/ \
    --pred_dir=/path/to/predictions/
