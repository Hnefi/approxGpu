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
    
    //dim3 nblocks(4,3);
    dim3 threadsPerBlock(32,32);

    // dynamically calculate how many thread blocks to launch
    int nBlocksWide = cols/32;
    if (cols % 32) nBlocksWide++;
    int nBlocksTall = rows/32;
    if (rows % 32) nBlocksTall++;
    dim3 nblocks(nBlocksWide,nBlocksTall);

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
    // dynamically calculate how many thread blocks to launch
    /*
    nBlocksWide = resizedCols/32;
    if (resizedCols % 32) nBlocksWide++;
    nBlocksTall = resizedRows/32;
    if (resizedRows % 32) nBlocksTall++;
    dim3 nBlocksResize(nBlocksWide,nBlocksTall);
    */

    cudaMalloc((void**)&resizeInt,rows*resizedCols*sizeof(float));
    cudaMalloc((void**)&dxInt,rows*cols*sizeof(float));
    cudaMalloc((void**)&dyInt,rows*cols*sizeof(float));
    cudaMalloc((void**)&resizeOutput,resizedRows*resizedCols*sizeof(float));
    cudaMalloc((void**)&dxOutput,rows*cols*sizeof(float));
    cudaMalloc((void**)&dyOutput,rows*cols*sizeof(float));

    // clear outputs since we only access some of these pixels, others are blank 
    cudaMemset(resizeOutput,0,resizedRows*resizedCols*sizeof(float));
    cudaMemset(dxOutput,0,rows*cols*sizeof(float));
    cudaMemset(dyOutput,0,rows*cols*sizeof(float));
    cudaMemset(resizeInt,0,rows*resizedCols*sizeof(float));


    printf("Finished kernel blur.... Calling kernel resize with resized rows: %d, resized cols: %d, and (%dx%d) grid of thread blocks.\n",resizedRows,resizedCols,nBlocksWide,nBlocksTall);
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

    cudaDeviceSynchronize();
  
   
    // TEMPORARY COPY FOR DEBUG.
    cudaMemcpy((void*)&(retStruct->horizEdge->data[0]),resizeInt,rows*resizedCols*sizeof(float),cudaMemcpyDeviceToHost);
    /*
    for(int i=0;i<rows*resizedCols;i++){
        printf("Element # %d of GPU intermediate array is: %0.4f\n",i,retStruct->horizEdge->data[i]);
    }*/

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

