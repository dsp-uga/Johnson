# Neuron Finding

## Constrained Non-negative Matrix Factorization

### Environment Setup

1. Create conda environment by based on `environment.yml` offered in CNMF folder.
```
$ conda env create -f environments.yml -n cnmf python=3.6
$ source activate cnmf
```

2. Clone the sources package by [CaImAn](https://github.com/flatironinstitute/CaImAn), go into CaImAn repository, and set up `caiman` in the conda environment, then delete CaImAn repository.
```
$ git clone https://github.com/flatironinstitute/CaImAn
$ cd CaImAn
$ python setup.py install
$ rm -rf CaImAn
```

**Add `sudo` before the command if you encounter the permision problems.**
 

### Running the test

```
$ python cnmf_test.py
```
