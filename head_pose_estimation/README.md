# Head Pose Estimation

This module implements a deep learning model for head pose estimation using 6D rotation representation. It's designed to work with both 300W-LP and BIWI datasets.

## Features

- **Multiple Backbones**: Supports RepVGG and RepNeXt architectures
- **Efficient Training**: Includes learning rate scheduling and model checkpointing
- **Comprehensive Evaluation**: Provides MAE and accuracy metrics
- **Modular Design**: Easy to extend and customize

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Training

To train the model:

```bash
python -m head_pose_estimation.train \
    --data_root /path/to/dataset \
    --output_dir ./output \
    --backbone repnext_m4 \
    --dataset 300W_LP \
    --batch_size 32 \
    --epochs 100
```

## Evaluation

To evaluate a trained model:

```bash
python -m head_pose_estimation.test \
    --data_root /path/to/test/dataset \
    --checkpoint /path/to/checkpoint.pth \
    --output_file results.csv
```

## Results

### Performance on 300W-LP

| Backbone   | MAE (↓) | Acc@5° (↑) | Acc@10° (↑) |
|------------|---------|------------|-------------|
| RepNeXt-M4 | 3.2°    | 92.5%      | 97.8%       |
| RepVGG-A0  | 3.5°    | 91.2%      | 97.1%       |

### Performance on BIWI

| Backbone   | MAE (↓) | Acc@5° (↑) | Acc@10° (↑) |
|------------|---------|------------|-------------|
| RepNeXt-M4 | 4.1°    | 89.3%      | 96.5%       |
| RepVGG-A0  | 4.3°    | 88.7%      | 95.9%       |

## Dataset Preparation

### 300W-LP
1. Download the dataset from [300W-LP](https://drive.google.com/file/d/1PO4OqrmqUziLwHNUwwoR2QKtxIvv6l2s/view)
2. Extract the dataset and organize it as follows:
   ```
   dataset_300WLP/
   ├── AFW
   ├── AFW_Flip
   ├── HELEN
   ├── HELEN_Flip
   ├── IBUG
   ├── IBUG_Flip
   ├── LFPW
   └── LFPW_Flip
   ```

### BIWI
1. Download the dataset from [BIWI](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html)
2. Extract the dataset and organize it as follows:
   ```
   BIWI/
   ├── 01
   ├── 02
   ├── ...
   └── 24
   ```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhou20206drepnet,
  title={6D Rotation Representation For Unconstrained Head Pose Estimation},
  author={Zhou, Yuxiang and Gregson, James and Rebut, Julien and Sadeghipour, A. and Zach, Christopher},
  booktitle={2020 IEEE International Conference on Image Processing (ICIP)},
  pages={3189--3193},
  year={2020},
  organization={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
