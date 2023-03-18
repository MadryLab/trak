============
Installation FAQs
============

How to install :code:`TRAK`?
----------------------------

The standard version of our package can be installed using

.. code-block:: bash

    pip install traker

To install the version of our package which contains a fast, custom CUDA kernel
for the JL projection step, use

.. code-block:: bash

    pip install traker[fast]

:code:`pip` will compile our CUDA kernel on your machine. For this to happen, you
need to have compatible versions of :code:`gcc` and :code:`CUDA toolkit`. See
the sections below for tips regarding this.

How to install :code:`nvcc` (:code:`CUDA toolkit`)?
---------------------------------------------------

Version required: CUDA >= 10.0

Instructions (Pick one option):

    Some machine might already have been setup with the coda toolkit. You can run nvcc in a terminal and check if it already exists. If you have a compatible version then you can proceed with the installation
    If you are logged in an unversity/company shared cluster, there is most of the time a way to enable/load a version of cuda tookit without having to install it. On clusters using modulefile, the command module avail will show you what is available to you. When in doubt, plese refer to the maintainers/documentation of your cluster
    Using conda: conda install -c conda-forge cudatoolkit-dev
    If you are root on your machine or feel confident with configuring the installation you can follow Nvidia instructions: https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html

How to install :code:`gcc`?
---------------------------

Version required:

    CUDA 11: 7.5 <= version <= 10
    CUDA 12: version >= 7.5

Instructions (Pick one option):

    Most Operating System come with gcc preinstalled. You can run gcc --version in a terminal to check if it's the case on your machine and which version you have. If you have a compatible version then you can proceed with the installation
    Using conda
        Install gcc and g++: conda install gcc_linux-64==9.3.0 gxx_linux-64=9.3.0
        Enable the compiler before runing pip install: export CXX=x86_64-conda-linux-gnu-g++ CC=x86_64-conda-linux-gnu-gcc. This has to be done in the same terminal
    If your operating ships with an incompatible compiler they usually let you install other version alongside what comes by default. Here is an example for ubuntu and gcc 10:
        Add repository: sudo add-apt-repository ppa:ubuntu-toolchain-r/test
        Update list of packages: sudo apt update
        Download/install gcc 10: sudo apt install gcc-10 g++-10
        Enable the compiler before runing pip install: export CXX=g++10 CC=gcc-10. This has to be done in the same terminal
