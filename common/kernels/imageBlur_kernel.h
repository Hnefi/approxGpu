__global__ void blurKernel_st1(int* inputPixels, float* intermediate, int* weightedKernel,float* hashes, float* threadReads,uint width, uint height,cudaTextureObject_t tref/*, other arguments */);
