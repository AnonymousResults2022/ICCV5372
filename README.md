CVPR1072

## Installation (Our codes are heavily borrowed from mmdet3d)
Please refer to [getting_started.md](docs/en/getting_started.md) for installation.

## Evaluation

### CP-Voxel-XXS 49.40% (mAP) 57.92% (NDS)
> ./tools/dist_test.sh configs/centerpoint/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_xxs.py checkpoints/xxs_epoch_20.pth 1 --eval bbox

### CP-Voxel-XS 54.00% (mAP) 62.06% (NDS)
> ./tools/dist_test.sh configs/centerpoint/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_xs.py checkpoints/xs_epoch_20.pth 1 --eval bbox

### Checkpoint files can be fetched in 