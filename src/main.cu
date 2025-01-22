/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <nppi_geometry_transforms.h>
#include <opencv2/opencv.hpp>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>
#include <cudnn.h>

#include <helper_cuda.h>
#include <helper_string.h>

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdint.h>


#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

__global__ void
addNoise(uint64_t seed, unsigned char *data, int size)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < size) {

    curandState rgnState;
    curand_init(seed, i, 0, &rgnState);
    const float rndRange = 30.0;
    float val = data[i] + curand_uniform(&rgnState) * 2.0 * rndRange - rndRange;
    if (val < 0) val = 0;
    if (val > 255) val = 255;
    data[i] = (unsigned char)val;
  }
}

inline int cudaDeviceInit(int argc, const char **argv) {
  int deviceCount;
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
    std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
    exit(EXIT_FAILURE);
  }

  int dev = findCudaDevice(argc, argv);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name
            << std::endl;

  checkCudaErrors(cudaSetDevice(dev));

  return dev;
}

bool printfNPPinfo(int argc, char *argv[]) {
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}

cv::Mat loadImage(const char* fileName) {
  cv::Mat img = cv::imread(fileName, cv::IMREAD_GRAYSCALE);
  img.convertTo(img, CV_32FC1);  
  return img;
}

void save_image(const char* fileName,
                float* buffer,
                int height,
                int width) {
  cv::Mat output_image(height, width, CV_32FC1, buffer);
  output_image.convertTo(output_image, CV_8UC1);  
  cv::imwrite(fileName, output_image);
}

int main(int argc, char *argv[]) {
  printf("%s Starting...\n\n", argv[0]);

  try {
    std::string sFilename;
    char *filePath = NULL;

    cudaDeviceInit(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false) {
      exit(EXIT_SUCCESS);
    }

    // Input file name
    if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
      getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
    } else {
      filePath = sdkFindFilePath("lena.pgm", argv[0]);
    }
    if (filePath) {
      sFilename = filePath;
    } else {
      sFilename = "lena.pgm";
    }

    // Output file name.
    std::string sNoisetFilename = sFilename;
    std::string::size_type dot = sNoisetFilename.rfind('.');
    if (dot != std::string::npos) {
      sNoisetFilename = sNoisetFilename.substr(0, dot);   
    }
    std::string sBoxFilename = sNoisetFilename;
    std::string sBoxCUDNNFilename = sNoisetFilename;
    sNoisetFilename += "_noisy.pgm";
    sBoxFilename += "_box_npp.pgm";
    sBoxCUDNNFilename += "_box_cudnn.png";

    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image
    npp::loadImage(sFilename, oHostSrc);

    // declare a device image and copy construct from the host image, i.e. upload host to device
    npp::ImageNPP_8u_C1 oDevice(oHostSrc);

    int totalImgSize = oDevice.height() * oDevice.pitch();
    int threadsPerBlock = 256;
    int blocksPerGrid =(totalImgSize + threadsPerBlock - 1) / threadsPerBlock;    
    addNoise<<<blocksPerGrid, threadsPerBlock>>>(time(NULL), oDevice.data(), totalImgSize);
    std::cout << "Cuda kernels with adding noise to an image" << std::endl;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHost(oDevice.size());
    // and copy the device result data into it
    oDevice.copyTo(oHost.data(), oHost.pitch());
    saveImage(sNoisetFilename, oHost);
    std::cout << "Saved image with noise: " << sNoisetFilename << std::endl;

    // size of the box filter
    const int boxSize = 5;
    const NppiSize  oMaskSize   = {boxSize, boxSize};
    const NppiPoint oMaskAchnor = {boxSize>>1, boxSize>>1};
    // compute maximal result image size
    const NppiSize  oSizeROI = {oDevice.width(), oDevice.height()};
    // allocate device box smoothed image
    npp::ImageNPP_8u_C1 oDeviceDst(oDevice.width(), oDevice.height());
    std::cout << "nppiFilterBox_8u_C1R " << std::endl;
    NPP_CHECK_NPP(nppiFilterBox_8u_C1R(oDevice.data(), oDevice.pitch(), oDeviceDst.data(), oDeviceDst.pitch(),
                                        oSizeROI, oMaskSize, oMaskAchnor));

    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHostBox(oDevice.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostBox.data(), oHostBox.pitch());
    saveImage(sBoxFilename, oHostBox);
    std::cout << "Saved image with NPPI based box smoothed noise: " << sBoxFilename << std::endl;

    nppiFree(oDevice.data());
    nppiFree(oDeviceDst.data());

    // load noisy image
    cv::Mat img = loadImage(sNoisetFilename.c_str());

    // CUDNN initialization
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cudnnTensorDescriptor_t inputDescriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&inputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(inputDescriptor,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/1,
                                          /*channels=*/1,
                                          /*image_height=*/img.rows,
                                          /*image_width=*/img.cols));

    cudnnFilterDescriptor_t kernelDescriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernelDescriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernelDescriptor,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*out_channels=*/1,
                                          /*in_channels=*/1,
                                          /*kernel_height=*/boxSize,
                                          /*kernel_width=*/boxSize));

    cudnnConvolutionDescriptor_t convDescriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDescriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDescriptor,
                                              /*pad_height=*/boxSize >> 1,
                                              /*pad_width=*/boxSize >> 1,
                                              /*vertical_stride=*/1,
                                              /*horizontal_stride=*/1,
                                              /*dilation_height=*/1,
                                              /*dilation_width=*/1,
                                              /*mode=*/CUDNN_CONVOLUTION,
                                              /*computeType=*/CUDNN_DATA_FLOAT));

    int batch_size{0}, channels{0}, height{0}, width{0};
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDescriptor,
                                                    inputDescriptor,
                                                    kernelDescriptor,
                                                    &batch_size,
                                                    &channels,
                                                    &height,
                                                    &width));

    std::cerr << "CUDNN Output Image: " << height << " x " << width << " x " << channels << std::endl;

    cudnnTensorDescriptor_t outputDescriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&outputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(outputDescriptor,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/1,
                                          /*channels=*/1,
                                          /*image_height=*/img.rows,
                                          /*image_width=*/img.cols));
   cudnnConvolutionFwdAlgoPerf_t *convAlgoPerf = new cudnnConvolutionFwdAlgoPerf_t();
   int returnedAlgoCount;
   checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
         cudnn,
         inputDescriptor,
         kernelDescriptor,
         convDescriptor,
         outputDescriptor,
         1,
         &returnedAlgoCount,
         convAlgoPerf));

   size_t workspaceBytes{0};
   checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
         cudnn,
         inputDescriptor,
         kernelDescriptor,
         convDescriptor,
         outputDescriptor,
         convAlgoPerf->algo,
         &workspaceBytes));

    void* workspace{nullptr};
    cudaMalloc(&workspace, workspaceBytes);

    // prepare input image buffers
    int imageBytes = batch_size * channels * height * width * sizeof(float);

    float* deviceInput{nullptr};
    cudaMalloc(&deviceInput, imageBytes);
    cudaMemcpy(deviceInput, img.ptr<float>(0), imageBytes, cudaMemcpyHostToDevice);

    float* deviceOutput{nullptr};
    cudaMalloc(&deviceOutput, imageBytes);
    cudaMemset(deviceOutput, 0, imageBytes);

    // prepare convolutional kernel
    float hostKernel[1][1][boxSize][boxSize];
    for (int kernel = 0; kernel < 1; ++kernel) {
      for (int channel = 0; channel < 1; ++channel) {
        for (int row = 0; row < boxSize; ++row) {
          for (int column = 0; column < boxSize; ++column) {
            hostKernel[kernel][channel][row][column] = 1.0/(boxSize * boxSize);
          }
        }
      }
    }
    float* deviceKernel{nullptr};
    cudaMalloc(&deviceKernel, sizeof(hostKernel));
    cudaMemcpy(deviceKernel, hostKernel, sizeof(hostKernel), cudaMemcpyHostToDevice);

    // run CUDNN convolution
    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                      &alpha,
                                      inputDescriptor,
                                      deviceInput,
                                      kernelDescriptor,
                                      deviceKernel,
                                      convDescriptor,
                                      convAlgoPerf->algo,
                                      workspace,
                                      workspaceBytes,
                                      &beta,
                                      outputDescriptor,
                                      deviceOutput));

    float* hostOutput = new float[imageBytes];
    cudaMemcpy(hostOutput, deviceOutput, imageBytes, cudaMemcpyDeviceToHost);

    save_image(sBoxCUDNNFilename.c_str(), hostOutput, height, width);
    std::cout << "Saved image with CUDNN based box smoothed noise: " << sBoxCUDNNFilename << std::endl;

    delete[] hostOutput;
    cudaFree(deviceKernel);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(inputDescriptor);
    cudnnDestroyTensorDescriptor(outputDescriptor);
    cudnnDestroyFilterDescriptor(kernelDescriptor);
    cudnnDestroyConvolutionDescriptor(convDescriptor);
    cudnnDestroy(cudnn);

    exit(EXIT_SUCCESS);
  } catch (npp::Exception &rException) {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  } catch (...) {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}
