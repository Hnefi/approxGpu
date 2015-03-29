__global__ void blurKernel_st1(int* inputPixels, int* intermediate, int* weightedKernel,int* hashes, int* threadReads,uint width, uint height,cudaTextureObject_t tref,int TEX/*, other arguments */);
