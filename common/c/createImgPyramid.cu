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
#include "../kernels/calcSobel_dX_kernel.h"
#include "../kernels/calcSobel_dY_kernel.h"

ImagePyramid* createImgPyramid(I2D* imageIn, cudaStream_t d_stream)
{
    int rows, cols;
    rows = imageIn->height;
    cols = imageIn->width;
  
    // setup kernels, thread objects, and GPU memory
    int weightedKernel[5] = {1,4,6,4,1};
    int sobelKernel_1[3] = {1,2,1};
    int sobelKernel_2[3] = {1,0,-1};
    
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
    int* d_weightedKernel,*sobel_kern_1,*sobel_kern_2;
    float* resizeInt, *dxInt, *dyInt, *dyInt_small, *dxInt_small;
    float* resizeOutput, *dxOutput, *dyOutput, *dxOutput_small, *dyOutput_small;

    //Pin host memory array for greatest speed transfer.
    HANDLE_ERROR( cudaHostRegister(&(imageIn->data[0]),rows*cols*sizeof(int),cudaHostRegisterPortable) );
    HANDLE_ERROR( cudaHostRegister(&weightedKernel[0],5*sizeof(int),cudaHostRegisterPortable) ) ;
    HANDLE_ERROR( cudaHostRegister(&sobelKernel_1[0],3*sizeof(int),cudaHostRegisterPortable) );
    HANDLE_ERROR( cudaHostRegister(&sobelKernel_2[0],3*sizeof(int),cudaHostRegisterPortable) );

    // SET UP MEMORY - local data
    cudaMalloc((void**)&(imageIn->d_weightedKernel),5*sizeof(int));
    cudaMalloc((void**)&(imageIn->sobel_kern_1),3*sizeof(int));
    cudaMalloc((void**)&(imageIn->sobel_kern_2),3*sizeof(int));
    d_weightedKernel = imageIn->d_weightedKernel;
    sobel_kern_1 = imageIn->sobel_kern_1;
    sobel_kern_2 = imageIn->sobel_kern_2;
    cudaMemcpyAsync(d_weightedKernel,&(weightedKernel[0]),5*sizeof(int),cudaMemcpyHostToDevice,d_stream);
    cudaMemcpyAsync(sobel_kern_1,&(sobelKernel_1[0]),3*sizeof(int),cudaMemcpyHostToDevice,d_stream);
    cudaMemcpyAsync(sobel_kern_2,&(sobelKernel_2[0]),3*sizeof(int),cudaMemcpyHostToDevice,d_stream);
    cudaStreamSynchronize(d_stream);

    // SET UP MEMORY
    cudaMalloc((void**)&(imageIn->d_inputPixels),rows*cols*sizeof(int));
    cudaMalloc((void**)&(imageIn->d_outputPixels),rows*cols*sizeof(float));
    cudaMalloc((void**)&(imageIn->d_intermediate),rows*cols*sizeof(float));
    cudaMalloc((void**)&(imageIn->resizeInt),rows*resizedCols*sizeof(float));
    cudaMalloc((void**)&(imageIn->dxInt),rows*cols*sizeof(float));
    cudaMalloc((void**)&(imageIn->dyInt),rows*cols*sizeof(float));
    cudaMalloc((void**)&(imageIn->resizeOutput),resizedRows*resizedCols*sizeof(float));
    cudaMalloc((void**)&(imageIn->dxOutput),rows*cols*sizeof(float));
    cudaMalloc((void**)&(imageIn->dyOutput),rows*cols*sizeof(float));

    cudaMalloc((void**)&(imageIn->dxOutput_small),resizedRows*resizedCols*sizeof(float));
    cudaMalloc((void**)&(imageIn->dyOutput_small),resizedRows*resizedCols*sizeof(float));
    cudaMalloc((void**)&(imageIn->dxInt_small),resizedRows*resizedCols*sizeof(float));
    cudaMalloc((void**)&(imageIn->dyInt_small),resizedRows*resizedCols*sizeof(float));
    
    d_inputPixels = imageIn->d_inputPixels;
    d_outputPixels = imageIn->d_outputPixels;
    d_intermediate = imageIn->d_intermediate;
    resizeInt = imageIn->resizeInt;
    dxInt = imageIn->dxInt;
    dyInt = imageIn->dyInt;
    dyInt_small = imageIn->dyInt_small;
    dxInt_small = imageIn->dxInt_small;
    resizeOutput = imageIn->resizeOutput;
    dxOutput = imageIn->dxOutput;
    dyOutput = imageIn->dyOutput;
    dxOutput_small = imageIn->dxOutput_small;
    dyOutput_small = imageIn->dyOutput_small;

    // Copy in input data and input kernels.
    cudaMemcpyAsync(d_inputPixels,&(imageIn->data[0]),rows*cols*sizeof(int),cudaMemcpyHostToDevice,d_stream);

    // clear outputs since we only access some of these pixels, others must be blank 
    cudaMemsetAsync(d_outputPixels,0,rows*cols*sizeof(float),d_stream);
    cudaMemsetAsync(d_intermediate,0,rows*cols*sizeof(float),d_stream);
    cudaMemsetAsync(resizeOutput,0,resizedRows*resizedCols*sizeof(float),d_stream);
    cudaMemsetAsync(resizeInt,0,rows*resizedCols*sizeof(float),d_stream);
    cudaMemsetAsync(dxOutput,0,rows*cols*sizeof(float),d_stream);
    cudaMemsetAsync(dyOutput,0,rows*cols*sizeof(float),d_stream);
    cudaMemsetAsync(dxInt,0,rows*cols*sizeof(float),d_stream);
    cudaMemsetAsync(dyInt,0,rows*cols*sizeof(float),d_stream);
    cudaMemsetAsync(dxOutput_small,0,resizedRows*resizedCols*sizeof(float),d_stream);
    cudaMemsetAsync(dyOutput_small,0,resizedRows*resizedCols*sizeof(float),d_stream);
    cudaMemsetAsync(dxInt_small,0,resizedRows*resizedCols*sizeof(float),d_stream);
    cudaMemsetAsync(dyInt_small,0,resizedRows*resizedCols*sizeof(float),d_stream);

    /* Kernel call */
    blurKernel_st1<<<nblocks,threadsPerBlock,0,d_stream>>>(d_inputPixels,d_intermediate,d_weightedKernel,cols,rows);
    blurKernel_st2<<<nblocks,threadsPerBlock,0,d_stream>>>(d_outputPixels,d_intermediate,d_weightedKernel,cols,rows);

    /* Call all kernels in one stream (order does not matter as they all read their input from d_outputPixels) */
    resizeKernel_st1<<<nblocks,threadsPerBlock,0,d_stream>>>(d_outputPixels,resizeInt,d_weightedKernel,rows,cols,resizedRows,resizedCols);
    resizeKernel_st2<<<nblocks,threadsPerBlock,0,d_stream>>>(resizeOutput,resizeInt,d_weightedKernel,rows,cols,resizedRows,resizedCols);

    /* Calc dX Sobel filter */
    calcSobel_dX_k1<<<nblocks,threadsPerBlock,0,d_stream>>>(d_outputPixels,dxInt,sobel_kern_1,sobel_kern_2,cols,rows);
    calcSobel_dX_k2<<<nblocks,threadsPerBlock,0,d_stream>>>(dxInt,dxOutput,sobel_kern_1,sobel_kern_2,cols,rows);

    calcSobel_dY_k1<<<nblocks,threadsPerBlock,0,d_stream>>>(d_outputPixels,dyInt,sobel_kern_1,sobel_kern_2,cols,rows);
    calcSobel_dY_k2<<<nblocks,threadsPerBlock,0,d_stream>>>(dyInt,dyOutput,sobel_kern_1,sobel_kern_2,cols,rows);

    /* Calc level 2 sobel filter (on resized images) */
    calcSobel_dX_k1<<<nblocks,threadsPerBlock,0,d_stream>>>(resizeOutput,dxInt_small,sobel_kern_1,sobel_kern_2,resizedCols,resizedRows);
    calcSobel_dX_k2<<<nblocks,threadsPerBlock,0,d_stream>>>(dxInt_small,dxOutput_small,sobel_kern_1,sobel_kern_2,resizedCols,resizedRows);

    calcSobel_dY_k1<<<nblocks,threadsPerBlock,0,d_stream>>>(resizeOutput,dyInt_small,sobel_kern_1,sobel_kern_2,resizedCols,resizedRows);
    calcSobel_dY_k2<<<nblocks,threadsPerBlock,0,d_stream>>>(dyInt_small,dyOutput_small,sobel_kern_1,sobel_kern_2,resizedCols,resizedRows);

    // deep copy into the destination F2D structures
    ImagePyramid* retStruct = (ImagePyramid*)malloc(sizeof(ImagePyramid));
    // alloc these sub-arrays as pinned memory (required for copyAsync)
    retStruct->blurredImg = fSetArray(rows,cols,0);
    retStruct->resizedImg = fSetArray(resizedRows,resizedCols,0);
    retStruct->horizEdge = fSetArray(rows,cols,0);
    retStruct->vertEdge = fSetArray(rows,cols,0);
    retStruct->horizEdge_small = fSetArray(resizedRows,resizedCols,0);
    retStruct->vertEdge_small = fSetArray(resizedRows,resizedCols,0);
    retStruct->tmp = fSetArray(rows,cols,0);
    HANDLE_ERROR( cudaHostRegister(&(retStruct->blurredImg->data[0]),rows*cols*sizeof(float),cudaHostRegisterPortable) );
    HANDLE_ERROR( cudaHostRegister(&(retStruct->resizedImg->data[0]),resizedRows*resizedCols*sizeof(float),cudaHostRegisterPortable) );
    HANDLE_ERROR( cudaHostRegister(&(retStruct->horizEdge->data[0]),rows*cols*sizeof(float),cudaHostRegisterPortable) );
    HANDLE_ERROR( cudaHostRegister(&(retStruct->vertEdge->data[0]),rows*cols*sizeof(float),cudaHostRegisterPortable) );
    HANDLE_ERROR( cudaHostRegister(&(retStruct->horizEdge_small->data[0]),resizedRows*resizedCols*sizeof(float),cudaHostRegisterPortable) );
    HANDLE_ERROR( cudaHostRegister(&(retStruct->vertEdge_small->data[0]),resizedRows*resizedCols*sizeof(float),cudaHostRegisterPortable) );

    cudaMemcpyAsync((void*)&(retStruct->blurredImg->data[0]),d_outputPixels,rows*cols*sizeof(float),cudaMemcpyDeviceToHost,d_stream);
    cudaMemcpyAsync((void*)&(retStruct->resizedImg->data[0]),resizeOutput,resizedRows*resizedCols*sizeof(float),cudaMemcpyDeviceToHost,d_stream);
    cudaMemcpyAsync((void*)&(retStruct->vertEdge->data[0]),dxOutput,rows*cols*sizeof(float),cudaMemcpyDeviceToHost,d_stream);   
    cudaMemcpyAsync((void*)&(retStruct->horizEdge->data[0]),dyOutput,rows*cols*sizeof(float),cudaMemcpyDeviceToHost,d_stream);   
    cudaMemcpyAsync((void*)&(retStruct->vertEdge_small->data[0]),dxOutput_small,resizedRows*resizedCols*sizeof(float),cudaMemcpyDeviceToHost,d_stream);   
    cudaMemcpyAsync((void*)&(retStruct->horizEdge_small->data[0]),dyOutput_small,resizedRows*resizedCols*sizeof(float),cudaMemcpyDeviceToHost,d_stream);   

    // UNSET Host memory pinning - local data
    cudaHostUnregister(&weightedKernel[0]);
    cudaHostUnregister(&sobelKernel_1[0]);
    cudaHostUnregister(&sobelKernel_2[0]);

    return retStruct;
}

void destroyImgPyramid(I2D* imageIn, ImagePyramid *retStruct)
{
    // UNSET Host memory pinning.
    cudaHostUnregister(&(imageIn->data[0]));

    cudaHostUnregister(&(retStruct->blurredImg->data[0]));
    cudaHostUnregister(&(retStruct->resizedImg->data[0]));
    cudaHostUnregister(&(retStruct->horizEdge->data[0]));
    cudaHostUnregister(&(retStruct->vertEdge->data[0]));
    cudaHostUnregister(&(retStruct->horizEdge_small->data[0]));
    cudaHostUnregister(&(retStruct->vertEdge_small->data[0]));
    cudaHostUnregister(&(retStruct->tmp->data[0]));

    cudaFree(imageIn->d_weightedKernel);
    cudaFree(imageIn->sobel_kern_1);
    cudaFree(imageIn->sobel_kern_2);
    cudaFree(imageIn->resizeInt);
    cudaFree(imageIn->dxInt);
    cudaFree(imageIn->dyInt);
    cudaFree(imageIn->resizeOutput);
    cudaFree(imageIn->dxOutput);
    cudaFree(imageIn->dyOutput);
    cudaFree(imageIn->d_inputPixels);
    cudaFree(imageIn->d_outputPixels);
    cudaFree(imageIn->d_intermediate);
    cudaFree(imageIn->dxInt_small);
    cudaFree(imageIn->dyInt_small);
    cudaFree(imageIn->dxOutput_small);
    cudaFree(imageIn->dyOutput_small);
}

