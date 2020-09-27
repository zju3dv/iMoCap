# Motion Capture from Internet Videos



> [Motion Capture from Internet Videos](https://arxiv.org/pdf/2008.07931.pdf)  
> Junting Dong*, Qing Shuai*, Yuanqing Zhang, Xian Liu, Xiaowei Zhou, Hujun Bao   
> ECCV 2020 
> [Project Page](https://zju3dv.github.io/iMoCap/)



## Datasets

### Internet video dataset

[Download](https://drive.google.com/file/d/1yD9VuKo5g4QDHAcBDli8a7absUVtZCXC/view?usp=sharing)

### Modified Human3.6M dataset 

You can download our modified Human3.6M dataset [here](https://drive.google.com/file/d/10KQTrp-TK6XvZuo27Xz0BPLF38S074co/view?usp=sharing).

### Create your own synthetic data

First, we split the origin videos into different folders, and store the 3D annotations as follows.

```
<path_to_data>
├── data_2d_h36m_cpn_ft_h36m_dbb.npz
├── joints3d
│   ├── S9_Directions 1.mat
│   ├── S9_Directions.mat
│   ├── ...
│   ├── ...
│   ├── ...
│   ├── S9_WalkTogether 1.mat
│   └── S9_WalkTogether.mat
└── S9
    ├── Directions
    │   ├── Directions.54138969.mp4
    │   ├── Directions.55011271.mp4
    │   ├── Directions.58860488.mp4
    │   └── Directions.60457274.mp4
    ├── Directions1
    │   ├── Directions1.54138969.mp4
    │   ├── Directions1.55011271.mp4
    │   ├── Directions1.58860488.mp4
    │   └── Directions1.60457274.mp4
    |   ......
    ├── WalkTogether
    │   ├── WalkTogether.54138969.mp4
    │   ├── WalkTogether.55011271.mp4
    │   ├── WalkTogether.58860488.mp4
    │   └── WalkTogether.60457274.mp4
    └── WalkTogether1
        ├── ......
```

We use finetune cpn output as our 2D pose from [videopose3d](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md)

```bash
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_cpn_ft_h36m_dbb.npz
```

After all, you can generate the synthetic data. More details can be found in the file `script/dataset/sample_h36m.py`.
```bash
python3 script/dataset/sample_h36m.py --video_path <path_to_data>/S9
```

## Quantitative evaluation

Our quantitative evaluation includes two parts: match and reconstruction. We provide the evaluation scripts as example.