#include <wb.h>

#define NUM_BINS 4096
#define BLOCK_SIZE 512 

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void 
gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

__global__ void 
histogram(unsigned int *input, unsigned int *bins, unsigned int num_elements,
          unsigned int num_bins)
{
	//@@ Write the kernel that computes the histogram
	//@@ Make sure to use the privitization technique
	//(hint: since NUM_BINS=4096 is larger than maximum allowed number of threads per block, 
	//be aware that threads would need to initialize more than one shared memory bin 
	//and update more than one global memory bin)
    __shared__ unsigned int private_histo[NUM_BINS];
    
    unsigned int i, binIdx;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (binIdx = threadIdx.x; binIdx < NUM_BINS; binIdx += blockDim.x)
        private_histo[binIdx] = 0u;

    __syncthreads();

    for (i = tid; i < num_elements; i += blockDim.x * gridDim.x)
        atomicAdd(&(private_histo[input[i]]), 1);

    __syncthreads();

    for (binIdx = threadIdx.x; binIdx < NUM_BINS; binIdx += blockDim.x)
        atomicAdd(&(bins[binIdx]), private_histo[binIdx]);
}

__global__ void 
saturate(unsigned int *bins, unsigned int num_bins)
{
	//@@ Write the kernel that applies saturtion to counters (i.e., if the bin value is more than 127, make it equal to 127)
    for (unsigned int j = 0; j < NUM_BINS / BLOCK_SIZE; j++)
    {
        if (bins[threadIdx.x + blockDim.x * j] > 127)
            bins[threadIdx.x + blockDim.x * j] = 127;
    }
}

int
main(int argc, char *argv[])
{
  wbArg_t args;
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength, "Integer");
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The number of bins is ", NUM_BINS);

  wbTime_start(GPU, "Allocating device memory");
  //@@ Allocate device memory here
  int binsSize = NUM_BINS * sizeof(float);
  int inputSize = inputLength * sizeof(float);

  cudaMalloc((void **) &deviceBins, binsSize);
  cudaMalloc((void **) &deviceInput, inputSize);
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating device memory");

  wbTime_start(GPU, "Copying input host memory to device");
  //@@ Copy input host memory to device
  cudaMemcpy(deviceInput, hostInput, inputSize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceBins, hostBins, binsSize, cudaMemcpyHostToDevice);
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input host memory to device");
	
  wbTime_start(GPU, "Clearing the bins on device");
  //@@ zero out the deviceBins using cudaMemset() 
  cudaMemset(deviceBins, 0, binsSize);
  wbTime_stop(GPU, "Clearing the bins on device");

  //@@ Initialize the grid and block dimensions here
  int numBlocks = ceil(inputLength / BLOCK_SIZE);
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  wbLog(TRACE, "Launching kernel");
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Invoke kernels: first call histogram kernel and then call saturate kernel
  histogram<<<dimGrid, dimBlock>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  saturate<<<dimGrid, dimBlock>>>(deviceBins, NUM_BINS);
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output device memory to host");
  //@@ Copy output device memory to host
  cudaMemcpy(hostInput, deviceInput, inputLength, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostBins, deviceBins, binsSize, cudaMemcpyDeviceToHost);
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output device memory to host");

  wbTime_start(GPU, "Freeing device memory");
  //@@ Free the device memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);
  wbTime_stop(GPU, "Freeing device memory");

  wbSolution(args, hostBins, NUM_BINS);

  free(hostBins);
  free(hostInput);
  return 0;
}
