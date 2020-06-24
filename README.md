# Env
This code has been tested with
- [HKU CS GPU Farm Phase 1](https://www.cs.hku.hk/gpu-farm/quickstart)
- Python 2.7
- TensorFlow 1.14
- CUDA 10.0
- Ubuntu 18.04.2 LTS

# KITTI

## raw
[raw_data_downloader.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data_downloader.zip)

```
raw_data

kitti_raw_stereo

kitti_raw_eigen

```


## pose
[data_odometry_color.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip)

[data_odometry_poses.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip)

```
GT_pose_snippets # generated

kitti_raw_odom # generated

odometry
    - dataset:
        - poses:
        - sequences:
```

## flow
[data_scene_flow_calib.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_calib.zip)

[data_scene_flow_multiview.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_multiview.zip)

[data_scene_flow_calib.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_calib.zip)

[data_scene_flow.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip)

```
flow:
    - data_scene_flow
    - data_scene_flow_calib
    - data_scene_flow_multiview
    - data_scene_flow_mv_dump_test_data # generated
```

# depth
[data_depth_annotated.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip)

[data_depth_selection.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip)

[data_depth_velodyne.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip)

```
depth:
    - depth_selection:
        - test_depth_completion_anonymous: 
            - image
            - intrinsics
            - velodyne_raw
        - test_depth_prediction_anonymous:
            - image
            - intrinsics
        - var_selection_cropped
            - groundtruth_depth
            - image
            - intrinsics
            - velodyne_raw
        - train
        - val
```