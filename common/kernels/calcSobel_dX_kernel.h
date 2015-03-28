__global__ void calcSobel_dX_k1(float* inputPixels, float* intermediate, float* hashes, float* threadReads,int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t t,int TEX);

__global__ void calcSobel_dX_k2(float* intermediate, float* outputPixels, int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t t,int TEX);
