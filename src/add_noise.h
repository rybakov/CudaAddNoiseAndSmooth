#pragma once

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>

__global__ void setup_kernel(curandState* state, uint64_t seed);

__global__ void addNoise(curandState* globalState, const unsigned char *data, int size);