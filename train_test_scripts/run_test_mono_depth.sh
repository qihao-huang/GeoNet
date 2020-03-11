PYTHON="/userhome/34/h3567721/anaconda3/envs/geonet-v/bin/python"

$PYTHON geonet_main.py \
    --mode=test_depth \
    --dataset_dir=/path/to/kitti/raw/dataset/ \
    --init_ckpt_file=/path/to/trained/model/ \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=/path/to/save/predictions/

$PYTHON kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=/path/to/kitti/raw/dataset/ \
    --pred_file=/path/to/predictions/