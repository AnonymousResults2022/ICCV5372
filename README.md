# ICCV5372


## Installation (Our codes are heavily borrowed from mmdet3d)
Please refer to [getting_started.md](docs/en/getting_started.md) for installation.

## Evaluation

### CP-Voxel-XXS 49.40% (mAP) 57.92% (NDS)
> ./tools/dist_test.sh configs/centerpoint/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_xxs.py checkpoints/xxs_epoch_20.pth 1 --eval bbox

### CP-Voxel-XS 54.00% (mAP) 62.06% (NDS)
> ./tools/dist_test.sh configs/centerpoint/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_xs.py checkpoints/xs_epoch_20.pth 1 --eval bbox

### Newly added Figures and Algorithm can be found in
![Histogram with RDD on KITTI.](https://github.com/AnonymousResults2022/CVPR1074/blob/main/Fig2_KITTI_rdd.jpeg)
![Histogram with RDD on nuScenes.](https://github.com/AnonymousResults2022/CVPR1074/blob/main/Fig2_nuscenes_rdd.jpeg)
![Visualization of RDD like Fig1.](https://github.com/AnonymousResults2022/CVPR1074/blob/main/Fig1_rdd.jpeg)
![Algorithm.](https://github.com/AnonymousResults2022/CVPR1074/blob/main/algorithm.jpeg)

