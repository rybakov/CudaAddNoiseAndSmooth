# Image denoisyng using box convolution based on NPP and CUDNN

## Overview

This project demonstrates the use of curand with CUDA for adding noise to an image.
Then it shows how to use NVIDIA Performance Primitives (NPP) to perform image denoising using box filter.
And it sjhows how to use CUDNN to perform image denoising using convolution with pre-specified kernel (we generate conv kernel with box shape).
The goal is to utilize GPU acceleration to efficiently add noise to an image, and denoise it using different CUDA libs.
The project is a part of the CUDA Advanced Libraries course.

## Code Organization

```bin/```
This folder should hold all binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder contain input image lena.pgm.
Output image with added noise lena_noisy.pgm.
Output image with smoothed noise by box filter from NPP: lena_box_npp.pgm
Output image with smoothed noise by convolutional box filter from CUDNN: lena_box_cudnn.png


```src/```
The source code should be placed here in a hierarchical fashion, as appropriate.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
This file should hold the human-readable set of instructions for installing the code so that it can be executed. If possible it should be organized around different operating systems, so that it can be done by as many people as possible with different constraints.

```Makefile or CMAkeLists.txt or build.sh```
There should be some rudimentary scripts for building your project's code in an automatic fashion.

```run.sh```
An optional script used to run your executable code, either with or without command-line arguments.

## Key Concepts

Performance Strategies, Image Processing, NPP, CUDNN Library

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, ppc64le, armv7l

## CUDA APIs involved

## Dependencies needed to build/run
FreeImage, CUDA, OpenCV, NPP, CUDNN

## Prerequisites

Download and install the [CUDA Toolkit 12.2](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## Build and Run


### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd CudaAddNoiseAndSmooth
$ make all
```


## Running the Program
After building the project, you can run the program using the following command:

```bash
Copy code
make run
```

This command will execute the compiled binary, rotating the input image (Lena.png) by 45 degrees, and save the result as Lena_rotated.png in the data/ directory.

If you wish to run the binary directly with custom input/output files and custom scale, you can use:

```bash
- Copy code
./bin/main --input data/lena.pgm
```

- Cleaning Up
To clean up the compiled binaries and other generated files, run:


```bash
- Copy code
make clean
```

This will remove all files in the bin/ directory.
