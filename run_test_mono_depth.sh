PYTHON="/userhome/34/h3567721/anaconda3/envs/geonet-v/bin/python"

$PYTHON geonet_main.py \
    --mode=test_depth \
    --dataset_dir="/userhome/34/h3567721/dataset/kitti/raw_data" \
    --init_ckpt_file=/path/to/trained/model/ \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=/path/to/save/predictions/

$PYTHON kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir="/userhome/34/h3567721/dataset/kitti/raw_data" \
    --pred_file=/path/to/predictions/