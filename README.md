## WildCamera
This repository contains the code for the paper: **[Tame a Wild Camera: In-the-Wild Monocular Camera Calibration](https://arxiv.org/abs/2306.10988)** in NeurIPS 2023.
<br>
Authors: [Shengjie Zhu](https://shngjz.github.io/), [Abhinav Kumar](https://sites.google.com/view/abhinavkumar), [Masa Hu](https://scholar.google.com/citations?user=Xs-NkFMAAAAJ&hl=en), and [Xiaoming Liu](https://www.cse.msu.edu/~liuxm/index2.html)
<br>
[[arXiv preprint]](https://arxiv.org/abs/2306.10988)  [[Prject Page]](https://shngjz.github.io/WildCamera.github.io/) [[Poster]](https://drive.google.com/file/d/1y8v0jBd6MFtP8urHNBCzh0wsK43djIj0/view?usp=sharing)

## Applications and Qualitative Results
- 4 DoF Camera Calibration (Zero-Shot)
  <details>

  -  Camera Calibration:

    https://github.com/ShngJZ/WildCamera/assets/128062217/748cf660-aebd-4a86-8d94-2be28650853b

  -  DollyZoom-Demo1:
    
    https://github.com/ShngJZ/WildCamera/assets/128062217/15b18902-9c18-460d-8b5e-7d728cbd63c0


  -  DollyZoom-Demo2:

    https://github.com/ShngJZ/WildCamera/assets/128062217/5722039d-d0c0-49db-a7a1-c83c5e69f7fd

  -  DollyZoom-Demo3:
    
    https://github.com/ShngJZ/WildCamera/assets/128062217/ef352b58-3e30-4b00-add8-6db5ae1d5de0

- Image Crop and Resize Detection and Restoration (Zero-Shot)
  <details>

  https://github.com/ShngJZ/WildCamera/assets/128062217/c390588f-63e2-4611-b546-b86946f3caf9
  
- In-the-Wild Monocular 3D Object Detection ([Omni3d](https://github.com/facebookresearch/omni3d))
  <details>

  https://github.com/ShngJZ/WildCamera/assets/128062217/d776e3d0-11c3-48c2-9a1b-e5adc10408ba

## Brief Introduction
<img src="asset/framework.png" width="1000" >
We calibrate 4 DoF intrinsic parameters for in-the-wild images.
The work systematically presents the connection between intrinsic and monocular 3D priors, e.g. intrinsic is inferrable from monocular depth and surface normals.
We additionally introduce an alternative monocular 3D prior, the incidence field, for calibration.

## Data Preparation
Pretrained models and data are held in [Hugging Face](https://huggingface.co/datasets/Shengjie/WildCamera/tree/main).
```
WildCamera
├── model_zoo
│   ├── Release
│   │   ├── wild_camera_all.pth
│   │   ├── wild_camera_gsv.pth
├── data
│   ├── MonoCalib
│   │   ├── ARKitScenes
│   │   ├── BIWIRGBDID
│   │   ├── CAD120
│   │   ├── ...
│   │   ├── Waymo
│   ├── UncalibTwoViewPoseEvaluation
│   │   ├── megadepth_test_1500
│   │   ├── scannet_test_1500
```
Use the script to download data in your preferred location. 
Entire dataset takes around 150 GB disk space.
```bash
./asset/download_wildcamera_dataset.sh
```

## Installation
```bash
conda create -n wildacamera python=3.8
conda activate wildacamera
conda install pytorch=1.10.0 torchvision cudatoolkit=11.1
pip install matplotlib, tqdm, timm, mmcv
```

## Demo
``` bash
# Download demo images
sh asset/download_demo_images.sh

# Estimate intrinsic over images collected from github
python demo/demo_inference.py

# Demo inference on dolly zoom videos
python demo/demo_dollyzoom.py
```

## Benchmark
``` bash
# Benchmark Tab.2 and Tab.4
python WildCamera/benchmark/benchmark_calibration.py --experiment_name in_the_wild

# Benchmark Tab.3
python WildCamera/benchmark/benchmark_calibration.py --experiment_name gsv

# Benchmark Tab.5
python WildCamera/benchmark/benchmark_crop.py

# Benchmark Tab.6
python WildCamera/benchmark/benchmark_uncalibtwoview_megadepth.py
python WildCamera/benchmark/benchmark_uncalibtwoview_scannet.py
```


## Citation <a name="citing"></a>

Please use the following BibTeX to cite our work.

```BibTeX
@inproceedings{zhu2023tame,
  author =       {Shengjie Zhu and Abhinav Kumar and Masa Hu and Xiaoming Liu},
  title =        {Tame a Wild Camera: In-the-Wild Monocular Camera Calibration},
  booktitle =    {NeurIPS},
  year =         {2023},
}
```


If you use the Tame-a-Wild-Camera benchmark, we kindly ask you to additionally cite all datasets. BibTex entries are provided below.

<details><summary>Dataset BibTex</summary>

```BibTex
@inproceedings{
  dehghan2021arkitscenes,
  title={{ARK}itScenes - A Diverse Real-World Dataset for 3D Indoor Scene Understanding Using Mobile {RGB}-D Data},
  author={Gilad Baruch and Zhuoyuan Chen and Afshin Dehghan and Tal Dimry and Yuri Feigin and Peter Fu and Thomas Gebauer and Brandon Joffe and Daniel Kurz and Arik Schwartz and Elad Shulman},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
  year={2021},
  url={https://openreview.net/forum?id=tjZjv_qh_CE}
}
```
```BibTex
@inproceedings{cordts2016cityscapes,
  title={The cityscapes dataset for semantic urban scene understanding},
  author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3213--3223},
  year={2016}
}
``` 
```BibTex
@inproceedings{geiger2012we,
  title={Are we ready for autonomous driving? the kitti vision benchmark suite},
  author={Geiger, Andreas and Lenz, Philip and Urtasun, Raquel},
  booktitle={2012 IEEE conference on computer vision and pattern recognition},
  pages={3354--3361},
  year={2012},
  organization={IEEE}
}
``` 
```BibTex
@inproceedings{li2018megadepth,
  title={Megadepth: Learning single-view depth prediction from internet photos},
  author={Li, Zhengqi and Snavely, Noah},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2041--2050},
  year={2018}
}
``` 
```BibTex
@inproceedings{yu2023mvimgnet,
  title={Mvimgnet: A large-scale dataset of multi-view images},
  author={Yu, Xianggang and Xu, Mutian and Zhang, Yidan and Liu, Haolin and Ye, Chongjie and Wu, Yushuang and Yan, Zizheng and Zhu, Chenming and Xiong, Zhangyang and Liang, Tianyou and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9150--9161},
  year={2023}
}
``` 
```BibTex
@article{fuhrmann2014mve,
  title={Mve-a multi-view reconstruction environment.},
  author={Fuhrmann, Simon and Langguth, Fabian and Goesele, Michael},
  journal={GCH},
  volume={3},
  pages={4},
  year={2014}
}
``` 
```BibTex
@inproceedings{caesar2020nuscenes,
  title={nuscenes: A multimodal dataset for autonomous driving},
  author={Caesar, Holger and Bankiti, Varun and Lang, Alex H and Vora, Sourabh and Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and Baldan, Giancarlo and Beijbom, Oscar},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={11621--11631},
  year={2020}
}
``` 
```BibTex
@inproceedings{Silberman:ECCV12,
  author    = {Nathan Silberman, Derek Hoiem, Pushmeet Kohli and Rob Fergus},
  title     = {Indoor Segmentation and Support Inference from RGBD Images},
  booktitle = {ECCV},
  year      = {2012}
}
``` 
```BibTex
@inproceedings{ahmadyan2021objectron,
  title={Objectron: A large scale dataset of object-centric videos in the wild with pose annotations},
  author={Ahmadyan, Adel and Zhang, Liangkai and Ablavatski, Artsiom and Wei, Jianing and Grundmann, Matthias},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={7822--7831},
  year={2021}
}
``` 
```BibTex
@inproceedings{sturm2012benchmark,
  title={A benchmark for the evaluation of RGB-D SLAM systems},
  author={Sturm, J{\"u}rgen and Engelhard, Nikolas and Endres, Felix and Burgard, Wolfram and Cremers, Daniel},
  booktitle={2012 IEEE/RSJ international conference on intelligent robots and systems},
  pages={573--580},
  year={2012},
  organization={IEEE}
}
``` 
```BibTex
@inproceedings{dai2017scannet,
  title={Scannet: Richly-annotated 3d reconstructions of indoor scenes},
  author={Dai, Angela and Chang, Angel X and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5828--5839},
  year={2017}
}
``` 
```BibTex
@article{chang2015shapenet,
  title={Shapenet: An information-rich 3d model repository},
  author={Chang, Angel X and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and others},
  journal={arXiv preprint arXiv:1512.03012},
  year={2015}
}
``` 
```BibTex
@inproceedings{xiao2013sun3d,
  title={Sun3d: A database of big spaces reconstructed using sfm and object labels},
  author={Xiao, Jianxiong and Owens, Andrew and Torralba, Antonio},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={1625--1632},
  year={2013}
}
``` 
```BibTex
@inproceedings{sun2020scalability,
  title={Scalability in perception for autonomous driving: Waymo open dataset},
  author={Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and Chouard, Aurelien and Patnaik, Vijaysai and Tsui, Paul and Guo, James and Zhou, Yin and Chai, Yuning and Caine, Benjamin and others},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={2446--2454},
  year={2020}
}
``` 
</details>














