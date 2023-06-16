# WildCamera
Code and data for **Tame a Wild Camera: In-the-Wild Monocular Camera Calibration, Zhu et al, Arxiv 2023**

# Applications and Qualitative Results
- 4 DoF Camera Calibration
  <details>

  -  Camera Calibration:

    https://github.com/ShngJZ/WildCamera/assets/128062217/cbc78faf-7128-4850-80f0-fe157b0deb4e

  -  DollyZoom-Demo1:
    
    https://github.com/ShngJZ/WildCamera/assets/128062217/0c25c605-2785-413b-bd54-6067a43c8987

  -  DollyZoom-Demo2:
    
    https://github.com/ShngJZ/WildCamera/assets/128062217/c0709e39-3704-456a-8724-10f87e7555e0

  -  DollyZoom-Demo3:
    
    https://github.com/ShngJZ/WildCamera/assets/128062217/320ab8e7-5808-47d0-ab35-16297e6fb695

- Image Crop and Resize Detection and Restoration
  <details>

  https://github.com/ShngJZ/WildCamera/assets/128062217/2abe54ea-497d-4c12-aa17-0e8ca29b85aa
  
- In-the-Wild Monocular 3D Object Detection
  <details>

  https://github.com/ShngJZ/WildCamera/assets/128062217/c94dcf6d-5378-4a14-9f1c-ee7a966e2d2f

# Introduction
<img src="asset/framework.png" width="1000" >
In (a), our work focuses on monocular camera calibration for in-the-wild images.
We recover the intrinsic from monocular 3D-prior.
In (c) - (e), an estimated depthmap is converted to surface normal using a groundtruth and noisy intrinsic individually.
Noisy intrinsic distorts the point cloud, consequently leading to inaccurate surface normal.
Motivated by the observation, we develop a solver that utilizes the consistency between the two to recover the intrinsic.
However, the solution exhibits numerical instability.
We then propose to learn the incidence field as an alternative 3D monocular prior.
The incidence field is the collection of the pixel-wise incidence ray, which originates from a $\textcolor[RGB]{237, 28, 36}{\text{3D point}}$, targets at a $\textcolor[RGB]{57, 181, 74}{\text{2D pixel}}$, and crosses the camera origin, as shown in (b).
Similar to depthmap and normal, a noisy intrinsic leads to a noisy incidence field, as in (e).
By same motivation, we develop neural network to learn in-the-wild incidence field and develop a RANSAC algorithm to recover intrinsic from the estimated incidence field.

























