# Neuron Finding

## Constrained Non-negative Matrix Factorization

### Environment Setup

1. Clone the sources package by [CaImAn](https://github.com/flatironinstitute/CaImAn)

```
$ git clone https://github.com/flatironinstitute/CaImAn
```

2. Replace the `environment.yml` file by the file `environments.yml` offered in this repository. (Replace `opencv3` (only processes in python 3.5) by `pip/opencv-python` (processes in python 3.6))

3. Duplicate the whole `use_cases` directory to the path where you save your `cnmf_neuron.py` script.

4. Process the following steps to finish package `caiman` installation working in your machine. (`environment.yml` has specified the requirements of `keras 2.0.9` and `tensorflow 1.0.0`. Installing by conda might be covered by the root environment.)

```
$ conda env create -f environment_mac.yml -n cnmf python=3.6
$ source activate cnmf
$ python setup.py install
$ python setup.py build_ext -i
```
Add `sudo` before the command if you encounter the permision problems.

### Running the test

```
$ python cnmf_neuron.py
```
