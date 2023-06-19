# WildCamera
Code and data for **Tame a Wild Camera: In-the-Wild Monocular Camera Calibration, Zhu et al, Arxiv 2023**

# Applications and Qualitative Results
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

# Introduction
<img src="asset/framework.png" width="1000" >
Our work focuses on monocular camera calibration for in-the-wild images.
We propose to learn the incidence field as a monocular 3D prior.
The incidence field is the collection of the pixel-wise incidence ray, which originates from a 3D point, targets at a 2D pixel, and crosses the camera origin, as shown in (b).
We develop a neural network to learn in-the-wild incidence field and a RANSAC algorithm to recover intrinsic from the estimated incidence field.

# Experiments
- **In-the-Wild Monocular Camera Calibration**

  We benchmark in-the-wild monocular camera calibration performance. 
  Entry ''Synthetic'' randomly generates novel intrinsics with image resizing and cropping.
  Entry ''Ours + Assumption'' assumes 1DoF intrinsic, i.e., assuming central focal point and identical focal length.
  Baseline ''[Perspective](https://github.com/jinlinyi/PerspectiveFields)'' is a recent CVPR'23 work.\
  <img src="asset/comparisons-in-the-wild-calibration.png" height="200" >

- **Comparisons to Monocular Camera Calibration with Geometry**\
  <img src="asset/comparisons-calibration-with-geometry.png" height="60" >

- **Comparisons to Calibration with Object**\
  <img src="asset/comparisons-calibration-with-object.png" height="110" >

# Live Demo
We are wroking on a Hugging Face interface for an online demo.

# Code, Data, and Model
We will release upon publication.

























