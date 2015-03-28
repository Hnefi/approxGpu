__global__ void calcSobel_dY_k2(float* intermediate, float* outputPixels, 
                                    int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t t, int TEX);


__global__ void calcSobel_dY_k1(float* inputPixels, float* intermediate, 
                                    int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t t,int TEX);
