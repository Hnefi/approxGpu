__global__ void calcSobel_dY_k2(int* intermediate, int* outputPixels, 
                                    int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t t, int TEX);


__global__ void calcSobel_dY_k1(int* inputPixels, int* intermediate, 
                                    int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t t,int TEX);
