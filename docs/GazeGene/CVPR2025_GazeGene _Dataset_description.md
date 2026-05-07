    ---
license: cc-by-nc-sa-4.0
extra_gated_fields:
  Name: text
  Email (Please provide email with official domain such as corporate domains or educational domains): text
  Contry: country
  Affiliation: text
  An URL of official website/image verifying your institutional affiliation: text
  Supervisor Name (Enter None if you do not have a supervisor which means you are a permanent employee in the university or institution): text
  Supervisor Email (Enter None if you do not have a supervisor which means you are a permanent employee in the university or institution): text
  Purpose of Use: text
  I have read and agree to the license and additional terms and conditions: checkbox
extra_gated_prompt: >-
  The GazeGene Dataset and the accompanying software are under CC BY-NC-SA 4.0
  License with additional terms: https://phi-ai.buaa.edu.cn
task_categories:
- feature-extraction
- image-feature-extraction
tags:
- Gaze
- Gaze Estimation
- Eye Tracking
size_categories:
- 1M<n<10M
---

# GazeGene: Large-scale Synthetic Gaze Dataset with 3D Eyeball Annotations

**Yiwei Bao, Zhiming Wang, Feng Lu**

{baoyiwei, zy2306418, lufeng}@buaa.edu.cn

This work is presented by [Phi-A<sup>2</sup>I Lab](https://phi-ai.buaa.edu.cn/).


<img src="https://cdn-uploads.huggingface.co/production/uploads/67e3a1a5f01ee3e3ab850212/vhgIdoorvXqle1_xnoIQV.png" style="width: 70%; height: auto;" />

We propose the GazeGene dataset: a large-scale synthetic gaze estimation dataset providing over 1M images with 3D eyeball annotations and accurate gaze annotations.

## Abstract

Thanks to the introduction of large-scale datasets, deep-learning has become the mainstream approach for appearance-based gaze estimation problems.  However, current large-scale datasets contain annotation errors and provide only a single vector for gaze annotation, lacking key information such as 3D eyeball structures.  Limitations in annotation accuracy and variety have constrained the progress in research and development of deep-learning methods for appearance-based gaze-related tasks. In this paper, we present GazeGene, a new large-scale synthetic gaze dataset with photo-realistic samples. More importantly, GazeGene not only provides accurate gaze annotations, but also offers 3D annotations of vital eye structures such as the pupil, iris, eyeball, optical and visual axes for the first time. Experiments show that GazeGene achieves comparable quality and generalization ability with real-world datasets, even outperforms most existing datasets on high-resolution images. Furthermore, its 3D eyeball annotations expand the application of deep-learning methods on various gaze-related tasks, offering new insights into this field.

## Dataset Characteristics

GazeGene provides images from 56 subjects under multi-camera settings with 9 cameras, 2000 frames each, result in 1,008,000 images in total. GazeGene also provides smooth and large head pose and gaze distribution for generalization ability.

We recommend utilizing the first 46 subjects for training and the final 10 subjects for testing.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/67e3a1a5f01ee3e3ab850212/hL7Ut7HY4NDKNIcTF46Fi.png)

Furthermore, to guarantee the variety of the data distribution, we add extensive expression, blink and pupil size, lighting temperature, lighting intensity and lighting direction changes in our dataset:

<img src="https://cdn-uploads.huggingface.co/production/uploads/67e3a1a5f01ee3e3ab850212/-YWoBIGE3yBtpdQKJfyJl.png" style="width: 70%; height: auto;" />

Due to the high fidelity of Metahuman and carefully designed synthetic strategy, our dataset does not introduce extra domain gap compared to real-world dataset. GazeGene even achieves better generalization performance than most existing real-world datasets:

<img src="https://cdn-uploads.huggingface.co/production/uploads/67e3a1a5f01ee3e3ab850212/vaiMFUNl4n9VKWNnGW4G-.png" style="width: 70%; height: auto;" />

**We hope that the high quality and detailed 3D eye annotation in this dataset will facilitate further advancements of appearance-based gaze estimation technique.**

## File Structure

Cropped faces and corresponding annotations.  We add random translation and scaling as augmentation during face cropping. 

```
GazeGene_FaceCrops/
├── Example.py                              # Example codes of how to utlize GazeGene images and annotations
├── subject1/
├── ...
├── subject56/
| ├── labels/
| │ ├── gaze_label_camera0.pkl   # Gaze direction and head pose annotations
| │ └── complex_camera0.pkl      # Detailed 3D eye structure annotations
| ├── imgs/
| │ └── subject56_0000.jpg                  # Face crops(0001.jpg - NNNN.jpg)
| ├── camera_info.pkl                       # Camera calibration parameters
└─└── subject_label.pkl                       # Subject-specific parameters such as eyeball radius, kappa angles
```

## Annotations

**By default, any physically meaningful labels in the dataset are measured in centimeters.**

### Camera_info.pkl

Contains camera calibration parameters as a list of dictionaries. The content of this file is consistent for both versions (FaceCrops version and normalized version).

```python
[
    # Parameters of camera 0
    {
        'cam_id': int,                      # Camera id from 0 to 9
        'intrinsic_matrix': np.array(3x3),  # Camera intrinsics
        'R_mat': np.array(3x3),             # Rotation matrix of current camera under WCS
        'T_vec': np.array(3,)               # Translation vector in centimeters of current camera under WCS
    },
    # Parameters of camera 1
    {
        ...
    },
    ...
]
```

### subject_label.pkl

Contains subject specific parameters as a dictionary.

```python
{
    'ID': int,                     # Subject ID from 1 to 56
    'gender': str,                 # ['F', 'M'] refers to female and male
    'ethicity': str,               # ['B', 'Y', 'W'] refers to Black, Yellow and White
    'eyecenter_L': np.array(3,)    # Left eyeball center coordinates under HCS
    'eyecenter_R': np.array(3,)    # Right eyeball center coordinates under HCS
    'eyeball_radius': float,       # Eyeball radius
    'iris_radius': float,          # Iris radius
    'cornea_radius': float,        # Cornea radius
    'cornea2center': float,        # Distance from cornea center to eyeball center
    'UVRadius': float,             # Normalized relative pupil size
    'L_kappa': np.array(3,),       # Euler angles of left eye kappa
    'R_kappa': np.array(3,)        # Euler angles of right eye kappa
}
```



### gaze_label_camera[id].pkl

Annotations for gaze directions and head poses. 

```python
{
    'img_path': str,                     # Path to image, imgs/camera0/subject44_0000.jpg
    'gaze_C': np.array(2000x3),          # Unit gaze vector under CCS of head gaze
    'visual_axis_L': np.array(2000x3),   # Unit vector of left eye visual axis
    'visual_axis_R': np.array(2000x3),   # Unit vector of right eye visual axis
    'optic_axis_L': np.array(2000x3),    # Unit vector of left eye optic axis
    'optic_axis_R': np.array(2000x3),    # Unit vector of right eye optic axis
    'gaze_target': np.array(2000x3),     # 3D coordinate of gaze target under CCS
    'gaze_depth': np.array(2000,),       # Length of gaze vergence depth
    'head_R_mat': np.array(2000x3x3),    # Head pose rotation matrix under CCS
    'head_T_vec': np.array(2000x3)       # Head translation vector under CCS
}
```



### Complex_label_camera[id].pkl

3D eyeball annotations and other annotations.

```python
{
    'img_path': str,                          # Path to image, imgs/camera0/subject44_0000.jpg
    'eyeball_center_2D': np.array(2000x2x2),  # Pixel coordinates of eyeball centers. (frame, left/right, coordinates)
    'eyeball_center_3D': np.array(2000x2x3),  # 3D coordinates under CCS of eyeball cenaters.
    'pupil_center_2D': np.array(2000x2x2),    # Pixel coordinates of pupil centers. (frame, left/right, coordinates)
    'pupil_center_3D': np.array(2000x2x3),    # 3D coordinates under CCS of eyeball cenaters
    'iris_mesh_2D': np.array(2000x2x100x2),   # Pixel coordinates of iris contours with 100 landmarks
    'iris_mesh_3D': np.array(2000x2x100x3),   # 3D coordinate of iris contours with 100 landmarks
    'intrinsic_matrix_cropped': np.array(2000x3x3),   # intrinsic matrix corresponds to the cropped and scaled face image.
}
```

**For codes of common gaze utilities, normalization and SOTA methods, please refer to our survey paper '[Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark](https://phi-ai.buaa.edu.cn/Gazehub/)'.**

## Citation

```latex
@inproceedings{bao2025gazegene,
  title={GazeGene: Large-scale Synthetic Gaze Dataset with 3D Eyeball Annotations},
  author={Bao, Yiwei and Wang, Zhiming and Lu, Feng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## License

This dataset is under [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/) with additional conditions and terms. Please refer to the [complete license file](https://phi-ai.buaa.edu.cn/project/GazeGene/License.pdf).

## Debug

If you encounter `bad zipfile offset (local header sig)` error during unzipping, here is a possible solution:
```bash
zip -F GazeGene_FaceCrops.zip --out fixed.zip
unzip fixed.zip
```

