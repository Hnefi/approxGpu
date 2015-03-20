__global__ void blurKernel_st2(float* outputPixels,float* intermediate, int* weightedKernel,uint width, uint height,cudaTextureObject_t tref /*, other arguments */);
