#!/usr/bin/env python

from setuptools import setup

# CNMF version
version = '1.0.0'

required = open('requirements.txt').read().split('\n')

# Reference by jasminetstone repackage CaImAn packages
# Original cnmf package: https://github.com/flatironinstitute/CaImAn
setup(
    name='cnmf',
    version=version,
    description='repackaged constrained matrix factorization algorithm for analysis of calcium imaging data',
    author='jasminetstone',
    author_email='jasminetstone@gmail.com',
    url='https://github.com/jasminetstone/cnmf',
    packages=['cnmf'],
    install_requires=required,
    long_description='See ' + 'https://github.com/jasminetstone/cnmf',
    license='GNU'
)
