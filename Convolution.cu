#include <wb.h>

#define MASK_WIDTH 5
#define O_TILE_WIDTH 16
#define BLOCK_WIDTH O_TILE_WIDTH + (MASK_WIDTH - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))
#define TILE_SIZE O_TILE_WIDTH * O_TILE_WIDTH * 3

//@@ INSERT CODE HERE 
//implement the tiled 2D convolution kernel with adjustments for channels
//use shared memory to reduce the number of global accesses, 
//handle the boundary conditions when loading input list 
//elements into the shared memory
//clamp your output values

__global__ void 
convolution_2d(float *output, float *input, const float * __restrict__ M,
               int height, int width, int channels)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int depth = threadIdx.z;
    int mask_radius = MASK_WIDTH / 2;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    int row_i = row - ((MASK_WIDTH - 1) * blockIdx.y) - mask_radius;
    int col_i = col - ((MASK_WIDTH - 1) * blockIdx.x) - mask_radius;
    int i, j, idx, s_idx;
    float px, mask_value;

    __shared__ float ds_i[TILE_SIZE];

    s_idx = (ty * blockDim.y + tx) * channels + depth;
    idx = (row_i * width + col_i) * channels + depth;
    
    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width)
    	ds_i[s_idx] = input[idx];
    else
    	ds_i[s_idx] = 0.0f;
    
    __syncthreads();

    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width)
    {
        if (tx >= mask_radius && tx < (blockDim.x - mask_radius) && 
            ty >= mask_radius && ty < (blockDim.y - mask_radius))
        {
            float out = 0.0f;
            for (i = 0; i < MASK_WIDTH; i++)
            {
                for (j = 0; j < MASK_WIDTH; j++)
                {
                    int current_row = ty - mask_radius + i;
                    int current_col = tx - mask_radius + j;

                    px = ds_i[(current_row * blockDim.y + current_col) * channels + depth];
                    mask_value = M[i * MASK_WIDTH + j];
                    out += px * mask_value;
                }
            }
            output[(row_i * width + col_i) * channels + depth] = clamp(out);
        }
    }
}

int main(int argc, char *argv[]) {
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char *inputImageFile;
    char *inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    float *hostMaskData;
    float *deviceInputImageData;
    float *deviceOutputImageData;
    float *deviceMaskData;
  
    arg = wbArg_read(argc, argv); /* parse the input arguments */
  
    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile  = wbArg_getInputFile(arg, 1);
  
    inputImage   = wbImport(inputImageFile);
    hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);
  
    assert(maskRows == MASK_WIDTH);    /* mask height is fixed to 5 */
    assert(maskColumns == MASK_WIDTH); /* mask width is fixed to 5 */
  
    imageWidth    = wbImage_getWidth(inputImage);
    imageHeight   = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
  
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  
    hostInputImageData  = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
  
    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
  
    wbTime_start(GPU, "Doing GPU memory allocation");
    //@@ INSERT CODE HERE
    //allocate device memory
    int ioSize = imageWidth * imageHeight * imageChannels * sizeof(float);
    int maskSize = maskRows * maskColumns * sizeof(float);
  
    cudaMalloc((void **) &deviceInputImageData,  ioSize);
    cudaMalloc((void **) &deviceOutputImageData, ioSize);
    cudaMalloc((void **) &deviceMaskData, maskSize);
    wbTime_stop(GPU, "Doing GPU memory allocation");
  
    wbTime_start(Copy, "Copying data to the GPU");
    //@@ INSERT CODE HERE
    //copy host memory to device
    cudaMemcpy(deviceInputImageData,  hostInputImageData,  ioSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOutputImageData, hostOutputImageData, ioSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData, hostMaskData, maskSize, cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");
  
    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    //initialize thread block and kernel grid dimensions
    dim3 dimBlock(O_TILE_WIDTH, O_TILE_WIDTH, imageChannels); 
    dim3 dimGrid((imageWidth  - 1) / (O_TILE_WIDTH - MASK_WIDTH) + 1,
                 (imageHeight - 1) / (O_TILE_WIDTH - MASK_WIDTH) + 1,
                  1);
    //invoke CUDA kernel	
    convolution_2d<<<dimGrid, dimBlock>>>(
            deviceOutputImageData,
            deviceInputImageData, 
            deviceMaskData,
            imageHeight,
            imageWidth,
            imageChannels);
    wbTime_stop(Compute, "Doing the computation on the GPU");
  
    wbTime_start(Copy, "Copying data from the GPU");
    //@@ INSERT CODE HERE
    //copy results from device to host	
    cudaMemcpy(hostOutputImageData, deviceOutputImageData, ioSize, cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");
  
    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
  
    wbSolution(arg, outputImage);
  
    //@@ INSERT CODE HERE
    //deallocate device memory	
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);
    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);
  
    return 0;
}
