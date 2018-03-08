# Neuron Finding

This repository contains various algorithms implemented on neurons images segmentation which are completed on CSCI 8360, Data Science Practicum at the University of Georgia, Spring 2018.

This project uses the time series image datasets of neurons from [CodeNeuro](http://neurofinder.codeneuro.org/). Each folder of training and testing images is a single plan, and the images are numbered according to their temporal ordering. The neurons in the images will flicker on and off as calcium is added. In this repository, we are offering three main algorithms as follows using different packages to locate the neurons and segment them out from the surrounding image.

1. Non-negative Matrix Factorization by [thunder-extraction](https://github.com/thunder-project/thunder-extraction)
2. Convolutional Neural Network by [Unet](https://github.com/jakeret/tf_unet)
3. Constrained Non-negative Matrix Factorization by [CaImAn](https://github.com/flatironinstitute/CaImAn)


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Anaconda](https://www.anaconda.com/)

### Environment Setting

1. Create conda environment by based on `environment.yml` offered in this repository.
```
$ conda env create -f environments.yml -n neuron python=3.6
$ source activate neuron
```

2. (**for CNMF**)
Clone the sources package by [CaImAn](https://github.com/flatironinstitute/CaImAn), go into CaImAn repository, and set up `caiman` in the conda environment, then delete CaImAn repository.
```
$ git clone https://github.com/flatironinstitute/CaImAn
$ cd CaImAn
$ python setup.py install
$ rm -rf CaImAn
```
Add `sudo` before the command if you encounter the permission problems.


## Running the tests

```
python -m [option-name] [args-for-the-option]
```

##### Options
  - `ThunderNMF`: Running NMF by `thunder-extraction`
  - `UNET`: Running CNN by `unet`
  - `CNMF`: Running CNMF by `caiman`

Each module provides their own arguments. Use `help()` to know more details when running the algorithms.

## Evaluation

Based on the neurons coordinates, five related scores to determine the results will be generated as follows:

- **Recall**: (number of matched regions)/(number of ground-truth regions)
- **Precision**:  (number of matched regions)/(number of our regions)
- **Inclusion**: (number of intersecting pixels)/(number of total pixels in the ground-truth regions)
- **Exclusion**: (number of intersecting pixels)/(number of total pixels in our regions)
- **Combined**:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\text{Combined}&space;=&space;\frac{2(\text{Recall}\times\text{Precision})}{(\text{Recall}&plus;\text{Precision})}" title="\text{Combined} = \frac{2(\text{Recall}\times\text{Precision})}{(\text{Recall}+\text{Precision})}" />
</p>

## Test Results



## Authors
(Ordered alphabetically)

- **Ankita Joshi** - [AnkitaJo](https://github.com/AnkitaJo)
- **I-Huei Ho** - [melanieihuei](https://github.com/melanieihuei)
- **Jeremy Shi** - [whusym](https://github.com/whusym)

See the [CONTRIBUTORS](CONTRIBUTORS.md) file for details.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
