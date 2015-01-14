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
#include "../kernels/imageResize_kernel.h"
//#include "../kernels/sobel_dX.h"
//#include "../kernels/sobel_dY.h"


ImagePyramid* createImgPyramid(I2D* imageIn)
{
    int rows, cols;
    rows = imageIn->height;
    cols = imageIn->width;
  
    // setup kernels, thread objects, and GPU memory
    int weightedKernel[5] = {1,4,6,4,1};
    int sobelKernel_1[3] = {1,0,-1};
    int sobelKernel_2[3] = {1,2,1};
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

    /* Kernel call */
    weightedBlurKernel<<<nblocks,threadsPerBlock>>>(d_inputPixels,d_outputPixels,d_intermediate,d_weightedKernel,cols,rows);

    cudaDeviceSynchronize();

    float* outputPixels = (float*)malloc(rows*cols*sizeof(float));
    cudaMemcpy(outputPixels,d_outputPixels,rows*cols*sizeof(float),cudaMemcpyDeviceToHost);

    // set up memory for other 3 images. first blur output serves as input here.
    float* resizeInt, *dxInt, *dyInt;
    float* resizeOutput, *dxOutput, *dyOutput;

    // resize is smaller than the blurred
    int resizedRows = floor((rows+1)/2);
    int resizedCols = floor((cols+1)/2);

    cudaMalloc((void**)&resizeInt,rows*cols*sizeof(int));
    cudaMalloc((void**)&dxInt,rows*cols*sizeof(int));
    cudaMalloc((void**)&dyInt,rows*cols*sizeof(int));
    cudaMalloc((void**)&resizeOutput,resizedRows*resizedCols*sizeof(int));
    cudaMalloc((void**)&dxOutput,rows*cols*sizeof(int));
    cudaMalloc((void**)&dyOutput,rows*cols*sizeof(int));

    // clear outputs since we only access some of these pixels, others are blank 
    cudaMemset(resizeOutput,0,resizedRows*resizedCols*sizeof(float));
    cudaMemset(dxOutput,0,rows*cols*sizeof(float));
    cudaMemset(dyOutput,0,rows*cols*sizeof(float));

    /* Call all kernels in one stream (order does not matter as they all read their input from d_outputPixels) */
    imageResizeKernel<<<nblocks,threadsPerBlock>>>(d_outputPixels,resizeOutput,resizeInt,d_weightedKernel,rows,cols,resizedRows,resizedCols);
    //sobelXFilter<<<nblocks,threadsPerBlock>>>(d_outputPixels,dxOutput,dxInt,sobelKernel_1,sobelKernel_2,rows,cols);
    //sobelYFilter<<<nblocks,threadsPerBlock>>>(d_outputPixels,dyOutput,dyInt,sobelKernel_2,sobelKernel_1,rows, cols);

    // synch back after this stream
    cudaDeviceSynchronize();

    // deep copy into the destination F2D structures
    ImagePyramid* retStruct = (ImagePyramid*)malloc(sizeof(ImagePyramid));
    retStruct->blurredImg = fSetArray(rows,cols,0);
    retStruct->resizedImg = fSetArray(resizedRows,resizedCols,0);
    retStruct->horizEdge = fSetArray(rows,cols,0);
    retStruct->vertEdge = fSetArray(rows,cols,0);
    memcpy((void*)&(retStruct->blurredImg->data[0]),(void*)&outputPixels[0],rows*cols*sizeof(float));
    cudaMemcpy((void*)&(retStruct->resizedImg->data[0]),resizeOutput,resizedRows*resizedCols*sizeof(float),cudaMemcpyDeviceToHost);

    free(outputPixels); // on the host size, we copied it into the retStruct so we didn't lose the image
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

