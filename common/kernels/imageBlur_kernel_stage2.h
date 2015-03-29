__global__ void blurKernel_st2(int* outputPixels,int* intermediate, int* weightedKernel,uint width, uint height,cudaTextureObject_t tref,int TEX /*, other arguments */);
