[![Python Versions](https://img.shields.io/pypi/pyversions/statista.png)](https://img.shields.io/pypi/pyversions/statista)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/MAfarrag/earth2observe.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MAfarrag/earth2observe/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/MAfarrag/earth2observe.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MAfarrag/earth2observe/alerts/)


[![codecov](https://codecov.io/gh/Serapieum-of-alex/statista/branch/main/graph/badge.svg?token=GQKhcj2pFK)](https://codecov.io/gh/Serapieum-of-alex/statista)
![GitHub last commit](https://img.shields.io/github/last-commit/MAfarrag/statista)
![GitHub forks](https://img.shields.io/github/forks/MAfarrag/statista?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/MAfarrag/statista?style=social)


Current release info
====================

| Name | Downloads                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Version | Platforms |
| --- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-statista-green.svg)](https://anaconda.org/conda-forge/statista) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/statista.svg)](https://anaconda.org/conda-forge/statista) [![Downloads](https://pepy.tech/badge/statista)](https://pepy.tech/project/statista) [![Downloads](https://pepy.tech/badge/statista/month)](https://pepy.tech/project/statista)  [![Downloads](https://pepy.tech/badge/statista/week)](https://pepy.tech/project/statista)  ![PyPI - Downloads](https://img.shields.io/pypi/dd/statista?color=blue&style=flat-square) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/statista.svg)](https://anaconda.org/conda-forge/statista) [![PyPI version](https://badge.fury.io/py/statista.svg)](https://badge.fury.io/py/statista) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/statista/badges/version.svg)](https://anaconda.org/conda-forge/statista) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/statista.svg)](https://anaconda.org/conda-forge/statista) [![Join the chat at https://gitter.im/Hapi-Nile/Hapi](https://badges.gitter.im/Hapi-Nile/Hapi.svg)](https://gitter.im/Hapi-Nile/Hapi?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) |

statista - Statistics package
=====================================================================
**statista** is a statistics package

statista

Main Features
-------------
  - Statistical Distributions
    - GEV
    - GUMBL
    - Normal
    - Exponential
  - Parameter estimation methods
    - Lmoments
    - ML
    - MOM
  - One-at-time (O-A-T) Sensitivity analysis.
  - Sobol visualization
  - Statistical descriptors
  - Extreme value analysis


Installing statista
===============

Installing `statista` from the `conda-forge` channel can be achieved by:

```
conda install -c conda-forge statista
```

It is possible to list all of the versions of `statista` available on your platform with:

```
conda search statista --channel conda-forge
```

## Install from Github
to install the last development to time you can install the library from github
```
pip install git+https://github.com/MAfarrag/statista
```

## pip
to install the last release you can easly use pip
```
pip install statista==0.5.0
```

Quick start
===========

```
  >>> import statista
```

[other code samples](https://statista.readthedocs.io/en/latest/?badge=latest)
