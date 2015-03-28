__global__ void calcSobel_dY_k2_Precise(float* intermediate, float* outputPixels, 
                                    int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t t);


__global__ void calcSobel_dY_k1_Precise(float* inputPixels, float* intermediate, 
                                    int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t t);

__global__ void calcSobel_dX_k1_Precise(float* inputPixels, float* intermediate, float* hashes, float* threadReads,int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t t);

__global__ void calcSobel_dX_k2_Precise(float* intermediate, float* outputPixels, int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t t);

__global__ void blurKernel_st1_Precise(int* inputPixels, float* intermediate, int* weightedKernel,float* hashes, float* threadReads,uint width, uint height,cudaTextureObject_t tref/*, other arguments */);

__global__ void blurKernel_st2_Precise(float* outputPixels,float* intermediate, int* weightedKernel,uint width, uint height,cudaTextureObject_t tref/*, other arguments */);

__global__ void resizeKernel_st1_Precise(float* inputPixels,float* intermediate, int* weightedKernel,uint width, uint height,uint r,uint c,cudaTextureObject_t t/*, other arguments */);

__global__ void resizeKernel_st2_Precise(float* outputPixels,float* intermediate, int* weightedKernel,uint width, uint height,uint r,uint c,cudaTextureObject_t t/*, other arguments */);
