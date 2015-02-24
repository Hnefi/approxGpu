// Mark Sutherland, Josh San Miguel
//  - Univ of Toronto

// Calls fast GPU implementations to create the requested GPU images (blur, resize, and sobel X/Y).

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <fcntl.h>
#include <unistd.h>

#include "sdvbs_common.h"
#include "../kernels/imageBlur_kernel.h"
#include "../kernels/imageBlur_kernel_stage2.h"
#include "../kernels/imageResize_kernel.h"
#include "../kernels/imageResize_kernel_st2.h"
//#include "../kernels/sobel_dX.h"
//#include "../kernels/sobel_dY.h"


ImagePyramid* createImgPyramid(I2D* imageIn, cudaStream_t d_stream)
{
    int rows, cols;
    rows = imageIn->height;
    cols = imageIn->width;
  
    // setup kernels, thread objects, and GPU memory
    int weightedKernel[5] = {1,4,6,4,1};
    int sobelKernel_1[3] = {1,0,-1};
    int sobelKernel_2[3] = {1,2,1};
    
    //dim3 nblocks(4,3);
    dim3 threadsPerBlock(32,32);

    // dynamically calculate how many thread blocks to launch
    int rowsIn = floor((rows+1)/4);
    int colsIn = floor((cols+1)/4);

    int resizedRows = floor((rows+1)/2);
    int resizedCols = floor((cols+1)/2);

    int nBlocksWide = colsIn/32;
    if (colsIn % 32) nBlocksWide++;
    int nBlocksTall = rowsIn/32;
    if (rowsIn % 32) nBlocksTall++;
    dim3 nblocks(nBlocksWide,nBlocksTall);
    //printf("Calculated block dimensions as: %d x %d\n",nBlocksWide,nBlocksTall);

    int* d_inputPixels;
    float* d_outputPixels;
    float* d_intermediate;
    int* d_weightedKernel;
    cudaMalloc((void**)&d_inputPixels,rows*cols*sizeof(int));
    cudaMalloc((void**)&d_outputPixels,rows*cols*sizeof(float));
    cudaMalloc((void**)&d_intermediate,rows*cols*sizeof(float));
    cudaMalloc((void**)&d_weightedKernel,5*sizeof(int));

    cudaMemcpyAsync(d_inputPixels,&(imageIn->data[0]),rows*cols*sizeof(int),cudaMemcpyHostToDevice,d_stream);
    cudaMemcpyAsync(d_weightedKernel,&(weightedKernel[0]),5*sizeof(int),cudaMemcpyHostToDevice,d_stream);
    cudaMemsetAsync(d_outputPixels,0,rows*cols*sizeof(float),d_stream);
    cudaMemsetAsync(d_intermediate,0,rows*cols*sizeof(float),d_stream);

    // set up memory for other 3 images. first blur output serves as input here.
    float* resizeInt, *dxInt, *dyInt;
    float* resizeOutput, *dxOutput, *dyOutput;
    cudaMalloc((void**)&resizeInt,rows*resizedCols*sizeof(float));
    cudaMalloc((void**)&dxInt,rows*cols*sizeof(float));
    cudaMalloc((void**)&dyInt,rows*cols*sizeof(float));
    cudaMalloc((void**)&resizeOutput,resizedRows*resizedCols*sizeof(float));
    cudaMalloc((void**)&dxOutput,rows*cols*sizeof(float));
    cudaMalloc((void**)&dyOutput,rows*cols*sizeof(float));

    // clear outputs since we only access some of these pixels, others are blank 
    cudaMemsetAsync(resizeOutput,0,resizedRows*resizedCols*sizeof(float),d_stream);
    cudaMemsetAsync(resizeInt,0,rows*resizedCols*sizeof(float),d_stream);
    cudaMemsetAsync(dxOutput,0,rows*cols*sizeof(float),d_stream);
    cudaMemsetAsync(dyOutput,0,rows*cols*sizeof(float),d_stream);

    /* Kernel call */
    blurKernel_st1<<<nblocks,threadsPerBlock,0,d_stream>>>(d_inputPixels,d_intermediate,d_weightedKernel,cols,rows);
    blurKernel_st2<<<nblocks,threadsPerBlock,0,d_stream>>>(d_outputPixels,d_intermediate,d_weightedKernel,cols,rows);

    /* Call all kernels in one stream (order does not matter as they all read their input from d_outputPixels) */
    resizeKernel_st1<<<nblocks,threadsPerBlock,0,d_stream>>>(d_outputPixels,resizeInt,d_weightedKernel,rows,cols,resizedRows,resizedCols);
    resizeKernel_st2<<<nblocks,threadsPerBlock,0,d_stream>>>(resizeOutput,resizeInt,d_weightedKernel,rows,cols,resizedRows,resizedCols);
    //sobelXFilter<<<nblocks,threadsPerBlock,0,d_stream>>>(d_outputPixels,dxOutput,dxInt,sobelKernel_1,sobelKernel_2,rows,cols);
    //sobelYFilter<<<nblocks,threadsPerBlock,0,d_stream>>>(d_outputPixels,dyOutput,dyInt,sobelKernel_2,sobelKernel_1,rows, cols);

    // deep copy into the destination F2D structures
    ImagePyramid* retStruct = (ImagePyramid*)malloc(sizeof(ImagePyramid));
    retStruct->blurredImg = fSetArray(rows,cols,0);
    retStruct->resizedImg = fSetArray(resizedRows,resizedCols,0);
    retStruct->horizEdge = fSetArray(rows,resizedCols,0);
    retStruct->vertEdge = fSetArray(rows,cols,0);
    cudaMemcpyAsync((void*)&(retStruct->blurredImg->data[0]),d_outputPixels,rows*cols*sizeof(float),cudaMemcpyDeviceToHost,d_stream);
    cudaMemcpyAsync((void*)&(retStruct->resizedImg->data[0]),resizeOutput,resizedRows*resizedCols*sizeof(float),cudaMemcpyDeviceToHost,d_stream);
    
    // TEMPORARY COPY FOR DEBUG.
    //cudaMemcpyAsync((void*)&(retStruct->horizEdge->data[0]),resizeInt,rows*resizedCols*sizeof(float),cudaMemcpyDeviceToHost,d_stream);
    //cudaMemcpyAsync((void*)&(retStruct->vertEdge->data[0]),d_intermediate,rows*cols*sizeof(float),cudaMemcpyDeviceToHost,d_stream);

    cudaFree(resizeInt);
    cudaFree(dxInt);
    cudaFree(dyInt);
    cudaFree(resizeOutput);
    cudaFree(dxOutput);
    cudaFree(dyOutput);
    cudaFree(d_inputPixels);
    cudaFree(d_outputPixels);
    cudaFree(d_intermediate);
    cudaFree(d_weightedKernel);

    return retStruct;
}

