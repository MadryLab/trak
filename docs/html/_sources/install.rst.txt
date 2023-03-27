.. _install-instructions:

=================
Installation FAQs
=================

How to install :code:`TRAK`?
----------------------------

Our package is hosted on `PyPI <https://pypi.org/>`_. The standard version of
our package can be installed using

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

.. note::

    Version required: :code:`CUDA >= 10.0`.

Pick one option:

* Some machines might already have been setup with the :code:`CUDA toolkit`.
  You can run 

  .. code:: bash

        nvcc --version

  in a terminal and check if it already exists. If you have a compatible version
  then you can proceed with the :code:`TRAK` installation.

* If you are logged in an unversity/company shared cluster, there is most
  of the time a way to enable/load a version of :code:`CUDA tookit` without
  having to install it. On clusters using :code:`modulefile`, the command

  .. code:: bash

    module avail

  will show you what is available to you. When in doubt, plese refer to the
  maintainers/documentation of your cluster.

* Using :code:`conda`:
  
  .. code:: bash

    conda install -c conda-forge cudatoolkit-dev
  
  Note that the version of :code:`CUDA toolkit` on the :code:`conda` index may
  be outdated.

* If you are root on your machine or feel confident with configuring the
  installation you can follow Nvidia instructions:
  https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html.

How to install :code:`gcc`?
---------------------------

.. note::

    Version required: If you have :code:`CUDA 11`, you will need :code:`gcc`
    with version :code:`7.5 <= version <= 10`. For :code:`CUDA 12`, you will
    need :code:`gcc` with :code:`version >= 7.5`.

Pick one option:

* Most Operating System come with gcc preinstalled. You can run 
  
  .. code:: bash
    
    gcc --version

  in a terminal to check if it's the case on your machine and which version you
  have. If you have a compatible version then you can proceed with the :code:`TRAK` 
  installation.
* If your operating ships with an incompatible compiler they usually let you
  install other version alongside what comes by default. Here is an example for
  ubuntu and gcc 10:
  1. Add repository: 
   
  .. code:: bash
        
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test 

  2. Update list of packages:
    
  .. code:: bash
        
    sudo apt update
    
  3. Download/install :code:`gcc 10`:
    
  .. code:: bash
        
    sudo apt install gcc-10 g++-10
    
  4. Enable the compiler before runing :code:`pip install traker[fast]`:
    
  .. note::
        
    This has to be done in the same terminal.

  .. code:: bash

    export CXX=g++10 CC=gcc-10

Verify that the installation worked
-----------------------------------

You can quickly verify that :code:`TRAK` has been correctly installed by running
some of our tests, e.g.:

.. code-block:: bash

  python -m pytest -sv tests/test_rademacher.py

Note that you'll need the optional :code:`[tests]` dependencies to run the tests.

Misc Q&A
--------

**Q: I'm using** :code:`zsh` **and when running**
:code:`pip install traker[fast]`,
**I get**

.. code::

  zsh: no matches found: traker[fast]

A: Make sure to escape the square brackets, i.e. run

.. code:: bash

  pip install traker\[fast\]