# Neuron Finding

## Constrained Non-negative Matrix Factorization

### Environment Setup

1. Clone the sources package by [CaImAn](https://github.com/flatironinstitute/CaImAn)

```
$ git clone https://github.com/flatironinstitute/CaImAn
```

2. Replace the `environment.yml` file inside the CaImAn directory by the file `environments.yml` offered in this repository. 

- Replace `opencv3` (only processes in python 3.5) by `pip/opencv-python` (processes in python 3.6)
- Specify the requirements of `keras 2.0.9` and `tensorflow 1.0.0`. (Install by conda might be covered by the root environment.)

3. Duplicate the whole `use_cases` folder and everything under the folder to the path where you save your `cnmf_test.py` and `cnmf_process.py` script. Place your testing sets in the same path. You will have these directories and files in your path:

<p align="center">
<img src="https://github.com/dsp-uga/Johnson/blob/cnmf/CNMF/folder_preview.png" height="250"/>
</p>

4. Process the following steps to finish package `caiman` installation working in your machine.

```
$ cd CaImAn
$ conda env create -f environment_mac.yml -n cnmf python=3.6
$ source activate cnmf
$ python setup.py install
$ python setup.py build_ext -i
```
5. Notice that:
- Add `sudo` before the command if you encounter the permision problems.
- You are able to delete the **CaImAn** folder once you finish your installation. 

### Running the test

```
$ python cnmf_test.py
```
