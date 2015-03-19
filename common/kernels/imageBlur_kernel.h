__global__ void blurKernel_st1(int* inputPixels, float* intermediate, int* weightedKernel,float* hashes,float* threadReads,uint width, uint height /*, other arguments */);
__global__ void texMark(float* input, float* output, uint width, uint height,int shift );
