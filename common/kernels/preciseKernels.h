__global__ void calcSobel_dY_k2_Precise(int* intermediate, int* outputPixels, 
                                    int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t t);


__global__ void calcSobel_dY_k1_Precise(int* inputPixels, int* intermediate, 
                                    int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t t);

__global__ void calcSobel_dX_k1_Precise(int* inputPixels, int* intermediate, int* hashes, int* threadReads,int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t t);

__global__ void calcSobel_dX_k2_Precise(int* intermediate, int* outputPixels, int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t t);

__global__ void blurKernel_st1_Precise(int* inputPixels, int* intermediate, int* weightedKernel,int* hashes, int* threadReads,uint width, uint height,cudaTextureObject_t tref/*, other arguments */);

__global__ void blurKernel_st2_Precise(int* outputPixels,int* intermediate, int* weightedKernel,uint width, uint height,cudaTextureObject_t tref/*, other arguments */);

__global__ void resizeKernel_st1_Precise(int* inputPixels,int* intermediate, int* weightedKernel,uint width, uint height,uint r,uint c,cudaTextureObject_t t/*, other arguments */);

__global__ void resizeKernel_st2_Precise(int* outputPixels,int* intermediate, int* weightedKernel,uint width, uint height,uint r,uint c,cudaTextureObject_t t/*, other arguments */);
