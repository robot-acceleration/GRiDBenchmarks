# GRiDBenchmarks

Provides the benchmark experiments for the paper ["GRiD: GPU-Accelerated Rigid Body Dynamics with Analytical Gradients"](https://brianplancher.com/publication/GRiD/)
```
@misc{plancher2021grid,
      title={GRiD: GPU-Accelerated Rigid Body Dynamics with Analytical Gradients}, 
      author={Brian Plancher and Sabrina M. Neuman and Radhika Ghosal and Scott Kuindersma and Vijay Janapa Reddi},
      year={2021},
      eprint={2109.06976},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

GRiDBenchmarks uses our [GRiD](https://github.com/robot-acceleration/GRiD) library and benchmarks it against [Pinocchio](https://github.com/stack-of-tasks/pinocchio/tree/pinocchio3-preview)'s pinocchio3-preview branch.

**This package contains submodules make sure to run ```git submodule update --init --recursive```** after cloning!

## Usage and API:
+ To run benchmarking on the packages please first set the CPU frequency to the maximum with ```setCPU.sh``` then run:
  1) ```timePinocchio.py URDF_PATH``` to compile and run CPU timing. Note that this only needs to compile once and will therefore run faster for additional URDFs.
  2) ```timeGRiD.py URDF_PATH``` to generate, compile, and run GPU timing.
+ If you would like to ensure that both packages are equivalent for your ```URDF``` set the variable ```TEST_FOR_EQUIVALENCE = 1``` in ```uitl/experiment_helpers.h``` and re-run the benchmarking (make sure to delete the ```timePinocchio.exe``` file before and after doing this as it needs to be re-compiled). This will print out the computed values by both packages for your robot.

## Benchmark Results
All results were collected on a high-performance workstation with a 3.8GHz eight-core Intel Core i7-10700K CPU and a 1.44GHz NVIDIA GeForce GTX 3080 GPU running Ubuntu 20.04 and CUDA 11.4. For clean timing measurements on the CPU, we disabled TurboBoost and fixed the clock frequency to the maximum. Code was compiled with ```Clang 12``` and ```g++9.4```, and time was measured with the Linux system call ```clock_gettime()```, using ```CLOCK_MONOTONIC``` as the source.

When performing multiple computations of rigid body dynamics algorithms, GRiD provides as much as a 7.6x speedup over a state-of-the-art, multi-threaded CPU implementation, and maintains as much as a 2.6x speedup when accounting for I/O overhead. 

![Latency (including GPU I/O overhead) for N = 16, 32, 64, 128, and 256 computations of the gradient of forward dynamics for both the Pinocchio CPU baseline and the GRiD GPU library for various robot models (IIWA, HyQ, and Atlas). Overlayed is the speedup (or slowdown) of GRiD as compared to Pinocchio both in terms of pure computation and including I/O overhead.](imgs/benchmark_multi_fd_grad.png)

We also find that the GPU is able to scale to more complex robots and algorithms better than the CPU by taking advantage of fine-grained parallelism induced by independent robot limbs and the independent columns of gradient computations.

![The scaling of single computation latency from IIWA to HyQ and IIWA to Atlas for both the Pinocchio CPU baseline and the GRiD GPU library for various rigid body dynamics algorithms (ID = Inverse Dynamics, Minv = Direct Minv, FD = Forward Dynamics and ∇ indicates the gradient of that algorithm). We also plot the scaling of the robot's dof as a measure of their increased complexity.](imgs/benchmark_single_scaling.png)

Below you will also find the single computation latency in microseconds per algorithm and robot (ID = Inverse Dynamics, Minv = Direct Minv, FD = Forward Dynamics and ∇ indicates the gradient of that algorithm).

| Algorithm | IIWA - CPU | HyQ - CPU | Atlas - CPU | IIWA - GPU | HyQ - GPU | Atlas - GPU |
|-----------|------------|-----------|-------------|------------|-----------|-------------|
| ID        | 0.2        | 0.3       | 1.1         | 3.0        | 3.2       | 8.0         |
| Minv      | 0.5        | 0.7       | 3.2         | 5.2        | 5.6       | 17.4        |
| FD        | 0.9        | 1.2       | 5.3         | 7.7        | 6.9       | 22.4        |
| ∇ID       | 1.4        | 2.1       | 10.0        | 6.3        | 5.8       | 19.5        |
| ∇FD       | 2.5        | 3.8       | 22.7        | 12.9       | 11.0      | 42.1        |

## Instalation Instructions:
### Install Python Dependencies
In order to support the wrapped packages there are 4 required external packages ```beautifulsoup4, lxml, numpy, sympy``` which can be automatically installed by running:
```shell
pip3 install -r requirements.txt
```
### Install Dependencies for the Various Packages
```
sudo apt-get update
sudo apt-get -y install git curl build-essential libglib2.0-dev dkms xorg xorg-dev cpufrequtils net-tools linux-headers-$(uname -r) meld apt-transport-https cmake libboost-all-dev liburdfdom-dev doxygen libgtest-dev
```
### Download and Install CUDA 
Note: for Ubuntu 20.04 see [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) for other distros
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```
**Add the following to ```~/.bashrc```**
```
# CUDA
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export PATH="opt/nvidia/nsight-compute/:$PATH"
```
### Download and install Clang/LLVM
```
sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
```
### Download and Install the Eigen CPU Linear Algebra Libaray
```
cd ~/Downloads
wget -q https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.bz2
tar -xf eigen-3.3.9.tar.bz2
cd eigen*
mkdir build && cd build
cmake ..
sudo make install
```
**Add symlinks**
```
sudo ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen
sudo ln -s /usr/local/include/eigen3/unsupported /usr/local/include/unsupported
```
### Download and Install EigenPy
```
sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y robotpkg-py38-eigenpy
```
**Add the following to ```~/.bashrc```**
```
# EigenPy
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/openrobots/lib/python3.8/site-packages:$PYTHONPATH
export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH
export C_INCLUDE_PATH=/opt/openrobots/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/opt/openrobots/include:$CPLUS_INCLUDE_PATH
```
### Download and Install CPP AD (Codegen)
**First [CPP AD](https://coin-or.github.io/CppAD/doc/install.htm) (v2020.3 for Codegen)**
```
cd ~/Downloads
git clone https://github.com/coin-or/CppAD.git
cd CppAD
git fetch && git checkout -q 83e249ec7819224138f35aaba564e2b977fb0078
mkdir build && cd build
cmake ..
sudo make install
```
**Then [CPP AD-Codegen](https://github.com/joaoleal/CppADCodeGen)**
```
cd ~/Downloads
git clone https://github.com/joaoleal/CppADCodeGen.git CppADCodeGen
cd CppADCodeGen && mkdir build && cd build
cmake .. -DLLVM_VERSION=12
sudo make install
```
### Download and Install the Pinocchio CPU Rigid Body Dynamics Library
*Note: if you would like to build the python interface swap ```-DBUILD_PYTHON_INTERFACE=OFF``` for ```-DPYTHON_EXECUTABLE=/usr/bin/python3```*
```
cd ~/Downloads
git clone --recursive https://github.com/stack-of-tasks/pinocchio
cd pinocchio
git checkout -b pinocchio3-preview origin/pinocchio3-preview 
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_WITH_AUTODIFF_SUPPORT=ON -DBUILD_WITH_CODEGEN_SUPPORT=ON -DCMAKE_CXX_COMPILER=clang++-12 -DBUILD_PYTHON_INTERFACE=OFF
make -j4
sudo make install
```
**Add the following to ```~/.bashrc```**
```
# Pinnochio
export PATH=/usr/local/bin:$PATH
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/lib/python3.8/site-packages:$PYTHONPATH
export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH
```
