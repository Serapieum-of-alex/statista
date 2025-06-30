# Installation


##### dependencies

# Required dependencies

- Python (3.11 or later)
- `numpy <https://www.numpy.org/>`__ (2 or later)
- `pandas <https://pandas.pydata.org/>`__ (2 or later)
- `SciPy <https://scipy.org/>`__ (1.14 or later)
- `scikit-learn <https://scikit-learn.org/>`__ (1.5 or later)
- `matplotlib <https://matplotlib.org//>`__ (1.5 or later)

##### Stable release
Please install ``statista`` in a Virtual environment so that its requirements don't tamper with your system's python.

## conda
the easiest way to install ``statista`` is using ``conda`` package manager. ``statista`` is available in the
`conda-forge <https://conda-forge.org/>`_ channel. To install
you can use the following command:

+ ``conda install -c conda-forge statista``

If this works it will install `statista` with all dependencies including Python and numpy, scipy and scikit-learn
and you skip the rest of the installation instructions.


## Installing Python and gdal dependencies

The main dependencies for statista are an installation of Python 3.9+, and scipy

## Installing Python

For Python we recommend using the Anaconda Distribution for Python 3, which is available
for download from https://www.anaconda.com/download/. The installer gives the option to
add ``python`` to your ``PATH`` environment variable. We will assume in the instructions
below that it is available in the path, such that ``python``, ``pip``, and ``conda`` are
all available from the command line.

Note that there is no hard requirement specifically for Anaconda's Python, but often it
makes installation of required dependencies easier using the conda package manager.

## Install as a conda environment

The easiest and most robust way to install statista is by installing it in a separate
conda environment. In the root repository directory there is an ``environment.yml`` file.
This file lists all dependencies. Either use the ``environment.yml`` file from the main branch
(please note that the main branch can change rapidly and break functionality without warning),
or from one of the releases {release}.

Run this command to start installing all statista dependencies:

+ ``conda env create -f environment.yml``

This creates a new environment with the name ``statista``. To activate this environment in
a session, run:

+ ``conda activate statista``

For the installation of statista there are two options (from the Python Package Index (PyPI)
or from Github). To install a release of statista from the PyPI (available from release 2018.1):

+ ``pip install statista=={release}``


## From sources


The sources for statista can be downloaded from the `Github repo`_.

You can either clone the public repository:

```console

$ git clone git://github.com/Serapieum-of-alex/statista

```
Or download the `tarball`_:

```console

$ curl -OJL https://github.com/Serapieum-of-alex/statista/tarball/main

```
Once you have a copy of the source, you can install it with:

```console

$ python -m pip install .


```
.. _Github repo: https://github.com/Serapieum-of-alex/statista
.. _tarball: https://github.com/Serapieum-of-alex/statista/tarball/main


To install directly from GitHub (from the HEAD of the main branch):

+ ``pip install git+https://github.com/Serapieum-of-alex/statista.git``

or from Github from a specific release:

+ ``pip install git+https://github.com/Serapieum-of-alex/statista.git@{release}``

Now you should be able to start this environment's Python with ``python``, try
``import statista`` to see if the package is installed.


More details on how to work with conda environments can be found here:
https://conda.io/docs/user-guide/tasks/manage-environments.html


If you are planning to make changes and contribute to the development of statista, it is
best to make a git clone of the repository, and do a editable install in the location
of you clone. This will not move a copy to your Python installation directory, but
instead create a link in your Python installation pointing to the folder you installed
it from, such that any changes you make there are directly reflected in your install.

+ ``git clone https://github.com/Serapieum-of-alex/statista.git``
+ ``cd statista``
+ ``activate statista``
+ ``pip install -e .``

Alternatively, if you want to avoid using ``git`` and simply want to test the latest
version from the ``main`` branch, you can replace the first line with downloading
a zip archive from GitHub: https://github.com/Serapieum-of-alex/statista/archive/main.zip
`libraries.io <https://libraries.io/github/Serapieum-of-alex/statista>`_.

## Install using pip

Besides the recommended conda environment setup described above, you can also install
statista with ``pip``. For the more difficult to install Python dependencies, it is best to
use the conda package manager:

+ ``conda install numpy scipy scikit-learn matplotlib pandas loguru``


you can check `libraries.io <https://libraries.io/github/Serapieum-of-alex/statista>`_. to check versions of the libraries


Then install a release {release} of statista (available from release 2018.1) with pip:

+ ``pip install statista=={release}``


## Check if the installation is successful

To check it the install is successful, go to the examples directory and run the following command:

+ ``python -m statista.*******``