/********************************
Author: Sravanthi Kota Venkata
********************************/

// Mark Sutherland, Joshua San Miguel
//  - University of Toronto

// Call a fast GPU implementation to blur this image.

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <ctype.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <sys/time.h>

#include "sdvbs_common.h"
#include "../kernels/imageBlur_kernel.h"

F2D* imageBlur(I2D* imageIn)
{
    int rows, cols;

    rows = imageIn->height;
    cols = imageIn->width;
  
    int weightedKernel[5] = {1,4,6,4,1};
    dim3 nblocks(4,3);
    dim3 threadsPerBlock(32,32);
    int* d_inputPixels;
    float* d_outputPixels;
    float* d_intermediate;
    int* d_weightedKernel;
    cudaMalloc((void**)&d_inputPixels,rows*cols*sizeof(int));
    cudaMalloc((void**)&d_outputPixels,rows*cols*sizeof(float));
    cudaMalloc((void**)&d_intermediate,rows*cols*sizeof(float));
    cudaMalloc((void**)&d_weightedKernel,5*sizeof(int));
    cudaMemcpy(d_inputPixels,&(imageIn->data[0]),rows*cols*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightedKernel,&(weightedKernel[0]),5*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemset(d_outputPixels,0,rows*cols*sizeof(float));
    cudaMemset(d_intermediate,0,rows*cols*sizeof(float));

    /* Kernel call */
    weightedBlurKernel<<<nblocks,threadsPerBlock>>>(d_inputPixels,d_outputPixels,d_intermediate,d_weightedKernel,cols,rows);

    cudaThreadSynchronize();

    float* outputPixels = (float*)malloc(rows*cols*sizeof(float));
    cudaMemcpy(outputPixels,d_outputPixels,rows*cols*sizeof(float),cudaMemcpyDeviceToHost);

    // deep copy into imageOut structure
    F2D* imageOut = fSetArray(rows,cols,0);
    memcpy((void*)&(imageOut->data[0]),(void*)&outputPixels[0],rows*cols*sizeof(float));
    cudaFree(d_inputPixels);
    cudaFree(d_outputPixels);
    cudaFree(d_intermediate);
    cudaFree(d_weightedKernel);
    return imageOut;
#ifdef APPROXIMATE
    // approximate all of the image data. NOTE: NOT THE KERNEL
    LVA_FUNCTION(2 /* int */, &(imageIn->data[0]),&(imageIn->data[rows*cols]),1);
    LVA_FUNCTION(5/*float*/, &(tempOut->data[0]),&(tempOut->data[rows*cols]),1);
    LVA_FUNCTION(5/*float*/, &(imageOut->data[0]),&(imageOut->data[rows*cols]),1);
#endif
#ifdef APPROXIMATE
    LVA_FUNCTION_RM(2 /* int */ ,&(imageIn->data[0]),&(imageIn->data[rows*cols]),1);
    LVA_FUNCTION_RM(5/*float*/ ,&(tempOut->data[0]),&(tempOut->data[rows*cols]),1);
    LVA_FUNCTION_RM(5/*float*/ ,&(imageOut->data[0]),&(imageOut->data[rows*cols]),1);
#endif
}
             

