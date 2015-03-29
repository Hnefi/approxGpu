__global__ void calcSobel_dX_k1(int* inputPixels, int* intermediate, int* hashes, int* threadReads,int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t t,int TEX);

__global__ void calcSobel_dX_k2(int* intermediate, int* outputPixels, int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t t,int TEX);
