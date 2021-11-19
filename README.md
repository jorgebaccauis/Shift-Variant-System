# SCCD: Shift-Variant Color-Coded Diffractive Spectral Imaging System

### [Project Page](https://jorgebaccauis.github.io/) | [Paper](https://www.osapublishing.org/optica/fulltext.cfm?uri=optica-8-11-1424&id=464500)

[Henry Arguello](https://scholar.google.com/citations?hl=es&user=R7gjbGIAAAAJ), [Samuel Pinilla](https://scholar.google.com/citations?hl=es&user=yGayy7sAAAAJ), [Yifan (Evan) Peng](http://stanford.edu/~evanpeng/), [Hayato Ikoma](https://scholar.google.com/citations?hl=es&user=ls53M1kAAAAJ), [Jorge Bacca](https://scholar.google.com/citations?hl=es&user=I5f1HjEAAAAJ), and [Gordon Wetzstein](http://stanford.edu/~gordonwz/)

This repository contains the scripts associated with the paper "Shift-Variant Color-Coded Diffractive Spectral Imaging System".

## Abstract
State-of-the-art snapshot spectral imaging (SI) systems introduce color-coded apertures (CCA) into their setups to obtain a flexible spatial-spectral modulation, allowing spectral information to be reconstructed from a set of coded measurements. Besides the CCA, other optical elements, such as lenses, prisms, beam splitters, are usually employed making systems large and impractical. Recently, diffractive optical elements (DOEs) have partially replaced refractive lenses to drastically reduce the size of the SI devices. The sensing model of these systems is represented as a projection modeled by a spatially shift-invariant convolution between the unknown scene and a point spread function (PSF) at each spectral band. However, the height maps of the DOE are the only free parameters that offer changes in the spectral modulation, which causes the ill-posedness of the reconstruction to increase significantly. To overcome this challenge, our work explores the advantages of the spectral modulation of an optical setup composed of a DOE and a CCA. Specifically, the light is diffracted by the DOE and then filtered by the CCA, located close to the sensor. A shift-variant property of the proposed system is clearly evidenced, resulting in a different PSF for each pixel, where a symmetric structure constraint is imposed on the CCA to reduce the high number of resulting PSFs. Additionally, we jointly design the DOE and the CCA parameters with a fully differentiable image formation model using an end-to-end approach to minimize the deviation between the true and reconstructed image over a large set of images. Simulation shows that the proposed system improves the spectral reconstruction quality in up to 4dB compared with current state-of-the-art systems. Finally, experimental results with a fabricated prototype in indoor and outdoor scenes validated the proposed system, where it can recover up to 49 high fidelity spectral bands in the 420-660 nm. 

# Installation

List of libraries required to execute the code.:
- python = 3.7.7
- Tensorflow = 2.2
- Keras = 2.4.3
- numpy
- scipy
- matplotlib
- h5py = 2.10
- opencv = 4.10
- poppy = 0.91

All of them can be installed via `conda` (`anaconda`), e.g.
```
conda install jupyter
```
or using pip install and the required file.

# Data
This work uses the following dataset. Please download the datasets and store them it correctly in the corresponding dataset folder (Train/Test).
- *MNIST dataset*: Provided in the `dataset/MNIST` folder.
- [*ARAD hyperspectral dataset:*](https://competitions.codalab.org/competitions/22225) It contains 450 hyperspectral training images and 10 validation images. The dataset  is available on the [challenge track websites](https://competitions.codalab.org/competitions/22225). Note that registration is required to access data.
We augmented the input datasets by scaling them to two different resolutions (half and double) following [paper](https://arxiv.org/abs/1409.1556). 

## Structure of directories

| Directory  | Description  |
| :--------: | :----------- | 
| `Dataset` | Folder that contains the datasets | 
| `Models and Tools`    | `.py` files for the custumer models | 
| `Pretrained Model`    | `.h5` pretrained model file |

## Citation
If you find our work useful in your research, please cite:

```
@article{arguello2021shift,
title={Shift-variant color-coded diffractive spectral imaging system},
author={Arguello, Henry and Pinilla, Samuel and Peng, Yifan and Ikoma, Hayato and Bacca, Jorge and Wetzstein, Gordon},
journal={Optica},
volume={8},
number={11},
pages={1424--1434},
year={2021},
publisher={Optical Society of America}
}
```

## Contact
If you have any questions, please contact

* Henry Arguello, henarfu@uis.edu.co
* Samuel Pinilla, samuel.pinilla@correo.uis.edu.co
* Yifan (Evan) Peng, evanpeng@stanford.edu
* Hayato Ikoma, hi.hayato.ikoma@gmail.com
* Jorge Bacca, jorge.bacca1@correo.uis.edu.co
* Gordon Wetzstein, gordon.wetzstein@stanford.edu 
