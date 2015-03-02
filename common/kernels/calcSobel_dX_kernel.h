__global__ void calcSobel_dX_k1(float* inputPixels, float* intermediate, int* kernel_1,int* kernel_2, uint width, uint height);

__global__ void calcSobel_dX_k2(float* intermediate, float* outputPixels, int* kernel_1,int* kernel_2, uint width, uint height);
