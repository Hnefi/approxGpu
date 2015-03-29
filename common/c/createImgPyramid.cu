// Mark Sutherland, Josh San Miguel
//  - Univ of Toronto

// Calls fast GPU implementations to create the requested GPU images (blur, resize, and sobel X/Y).

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <ctype.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

#include "sdvbs_common.h"
#include "../kernels/imageBlur_kernel.h"
#include "../kernels/imageBlur_kernel_stage2.h"
#include "../kernels/imageResize_kernel.h"
#include "../kernels/imageResize_kernel_st2.h"
#include "../kernels/calcSobel_dX_kernel.h"
#include "../kernels/calcSobel_dY_kernel.h"
#include "../kernels/preciseKernels.h"

using std::cout;
using std::endl;

ImagePyramid* createImgPyramid(I2D* imageIn, cudaTextureObject_t* texObj, bool train_set,int loadsToReplace,bool precise)
{
    int rows, cols;
    rows = imageIn->height;
    cols = imageIn->width;
    cudaTextureObject_t objToKernel = *texObj;
  
    // setup kernels, thread objects, and GPU memory
    int weightedKernel[5] = {1,4,6,4,1};
    int sobelKernel_1[3] = {1,2,1};
    int sobelKernel_2[3] = {1,0,-1};
    
    //dim3 nblocks(4,3);
    dim3 threadsPerBlock(16,16);

    // dynamically calculate how many thread blocks to launch
    int rowsIn = floor((rows+1)/8);
    int colsIn = floor((cols+1)/8);

    int resizedRows = floor((rows+1)/2);
    int resizedCols = floor((cols+1)/2);

    int nBlocksWide = colsIn/16;
    if (colsIn % 16) nBlocksWide++;
    int nBlocksTall = rowsIn/16;
    if (rowsIn % 16) nBlocksTall++;
    dim3 nblocks(nBlocksWide,nBlocksTall);
    //printf("Calculated block dimensions as: %d x %d\n",nBlocksWide,nBlocksTall);

    int* d_inputPixels;
    int* d_outputPixels;
    int* d_origInput;
    int* d_origInput2;
    int* d_origInput3;
    int* d_intermediate;
    int* d_weightedKernel,*sobel_kern_1,*sobel_kern_2;
    int* resizeInt, *dxInt, *dyInt, *dyInt_small, *dxInt_small;
    int* resizeOutput, *dxOutput, *dyOutput, *dxOutput_small, *dyOutput_small;

    int* threadReads, *threadHashes;
    int* reads, *hashes;
    int bytesForSmem = 16*16 * 3 * sizeof(int); // each thread gets 3 entries of 4 bytes each
    if(train_set == true) {
        reads = (int*) calloc(5*rows*cols,sizeof(int));
        hashes = (int*) calloc(5*rows*cols,sizeof(int));
        HANDLE_ERROR( cudaMalloc((void**)&threadReads,5*rows*cols*sizeof(int)) );
        HANDLE_ERROR( cudaMalloc((void**)&threadHashes,5*rows*cols*sizeof(int)) );
        cudaMemset(threadReads,0xff,5*rows*cols*sizeof(int));
        cudaMemset(threadHashes,0xff,5*rows*cols*sizeof(int));
    }

    // SET UP MEMORY - local data
    cudaMalloc((void**)&(imageIn->d_weightedKernel),5*sizeof(int));
    cudaMalloc((void**)&(imageIn->sobel_kern_1),3*sizeof(int));
    cudaMalloc((void**)&(imageIn->sobel_kern_2),3*sizeof(int));
    d_weightedKernel = imageIn->d_weightedKernel;
    sobel_kern_1 = imageIn->sobel_kern_1;
    sobel_kern_2 = imageIn->sobel_kern_2;
    cudaMemcpy(d_weightedKernel,&(weightedKernel[0]),5*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(sobel_kern_1,&(sobelKernel_1[0]),3*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(sobel_kern_2,&(sobelKernel_2[0]),3*sizeof(int),cudaMemcpyHostToDevice);

    // SET UP MEMORY
    cudaMalloc((void**)&(imageIn->d_inputPixels),rows*cols*sizeof(int));
    cudaMalloc((void**)&(imageIn->d_outputPixels),rows*cols*sizeof(int));
    cudaMalloc((void**)&(imageIn->d_intermediate),rows*cols*sizeof(int));
    cudaMalloc((void**)&(imageIn->resizeInt),rows*resizedCols*sizeof(int));
    cudaMalloc((void**)&(imageIn->dxInt),rows*cols*sizeof(int));
    cudaMalloc((void**)&(imageIn->dyInt),rows*cols*sizeof(int));
    cudaMalloc((void**)&(imageIn->resizeOutput),resizedRows*resizedCols*sizeof(int));
    cudaMalloc((void**)&(imageIn->dxOutput),rows*cols*sizeof(int));
    cudaMalloc((void**)&(imageIn->dyOutput),rows*cols*sizeof(int));

    cudaMalloc((void**)&(imageIn->dxOutput_small),resizedRows*resizedCols*sizeof(int));
    cudaMalloc((void**)&(imageIn->dyOutput_small),resizedRows*resizedCols*sizeof(int));
    cudaMalloc((void**)&(imageIn->dxInt_small),resizedRows*resizedCols*sizeof(int));
    cudaMalloc((void**)&(imageIn->dyInt_small),resizedRows*resizedCols*sizeof(int));
    
    cudaMalloc((void**)&d_origInput,rows*cols*sizeof(int));
    cudaMalloc((void**)&d_origInput2,rows*cols*sizeof(int));
    cudaMalloc((void**)&d_origInput3,rows*cols*sizeof(int));

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


    I2D* origPixelInput = iDeepCopy(imageIn);

    // Copy in input data and input kernels.
    cudaMemcpy(d_inputPixels,&(imageIn->data[0]),rows*cols*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_origInput,&(origPixelInput->data),rows*cols*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_origInput2,&(origPixelInput->data),rows*cols*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_origInput3,&(origPixelInput->data),rows*cols*sizeof(int),cudaMemcpyHostToDevice);

    // clear outputs since we only access some of these pixels, others must be blank 
    cudaMemset(d_outputPixels,0,rows*cols*sizeof(int));
    cudaMemset(d_intermediate,0,rows*cols*sizeof(int));
    cudaMemset(resizeOutput,0,resizedRows*resizedCols*sizeof(int));
    cudaMemset(resizeInt,0,rows*resizedCols*sizeof(int));
    cudaMemset(dxOutput,0,rows*cols*sizeof(int));
    cudaMemset(dyOutput,0,rows*cols*sizeof(int));
    cudaMemset(dxInt,0,rows*cols*sizeof(int));
    cudaMemset(dyInt,0,rows*cols*sizeof(int));
    cudaMemset(dxOutput_small,0,resizedRows*resizedCols*sizeof(int));
    cudaMemset(dyOutput_small,0,resizedRows*resizedCols*sizeof(int));
    cudaMemset(dxInt_small,0,resizedRows*resizedCols*sizeof(int));
    cudaMemset(dyInt_small,0,resizedRows*resizedCols*sizeof(int));

    if(!precise) {
        blurKernel_st1<<<nblocks,threadsPerBlock,bytesForSmem>>>(d_inputPixels,d_intermediate,d_weightedKernel,threadHashes,threadReads,cols,rows,objToKernel,loadsToReplace*2);
        blurKernel_st2<<<nblocks,threadsPerBlock,bytesForSmem>>>(d_outputPixels,d_intermediate,d_weightedKernel,cols,rows,objToKernel,loadsToReplace*2);

        resizeKernel_st1<<<nblocks,threadsPerBlock,bytesForSmem>>>(d_origInput,resizeInt,d_weightedKernel,rows,cols,resizedRows,resizedCols,objToKernel,loadsToReplace*2);
        resizeKernel_st2<<<nblocks,threadsPerBlock,bytesForSmem>>>(resizeOutput,resizeInt,d_weightedKernel,rows,cols,resizedRows,resizedCols,objToKernel,loadsToReplace*2);

        //TODO: this is reading origInput rather than d_outputPixels
        calcSobel_dX_k1<<<nblocks,threadsPerBlock,bytesForSmem>>>(d_origInput2,dxInt,threadHashes,threadReads,sobel_kern_1,sobel_kern_2,cols,rows,objToKernel,loadsToReplace);
        calcSobel_dX_k2<<<nblocks,threadsPerBlock,bytesForSmem>>>(dxInt,dxOutput,sobel_kern_1,sobel_kern_2,cols,rows,objToKernel,loadsToReplace);

        calcSobel_dY_k1<<<nblocks,threadsPerBlock,bytesForSmem>>>(d_origInput3,dyInt,sobel_kern_1,sobel_kern_2,cols,rows,objToKernel,loadsToReplace);
        calcSobel_dY_k2<<<nblocks,threadsPerBlock,bytesForSmem>>>(dyInt,dyOutput,sobel_kern_1,sobel_kern_2,cols,rows,objToKernel,loadsToReplace);

        /*
        calcSobel_dX_k1<<<nblocks,threadsPerBlock,bytesForSmem>>>(resizeOutput,dxInt_small,threadHashes,threadReads,sobel_kern_1,sobel_kern_2,resizedCols,resizedRows,objToKernel,loadsToReplace);
        calcSobel_dX_k2<<<nblocks,threadsPerBlock,bytesForSmem>>>(dxInt_small,dxOutput_small,sobel_kern_1,sobel_kern_2,resizedCols,resizedRows,objToKernel,loadsToReplace);

        calcSobel_dY_k1<<<nblocks,threadsPerBlock,bytesForSmem>>>(resizeOutput,dyInt_small,sobel_kern_1,sobel_kern_2,resizedCols,resizedRows,objToKernel,loadsToReplace);
        calcSobel_dY_k2<<<nblocks,threadsPerBlock,bytesForSmem>>>(dyInt_small,dyOutput_small,sobel_kern_1,sobel_kern_2,resizedCols,resizedRows,objToKernel,loadsToReplace);
        */
    } else {
        blurKernel_st1_Precise<<<nblocks,threadsPerBlock,bytesForSmem>>>(d_inputPixels,d_intermediate,d_weightedKernel,threadHashes,threadReads,cols,rows,objToKernel);
        blurKernel_st2_Precise<<<nblocks,threadsPerBlock,bytesForSmem>>>(d_outputPixels,d_intermediate,d_weightedKernel,cols,rows,objToKernel);

        resizeKernel_st1_Precise<<<nblocks,threadsPerBlock,bytesForSmem>>>(d_origInput,resizeInt,d_weightedKernel,rows,cols,resizedRows,resizedCols,objToKernel);
        resizeKernel_st2_Precise<<<nblocks,threadsPerBlock,bytesForSmem>>>(resizeOutput,resizeInt,d_weightedKernel,rows,cols,resizedRows,resizedCols,objToKernel);

        //TODO: this is reading origInput rather than d_outputPixels
        calcSobel_dX_k1_Precise<<<nblocks,threadsPerBlock,bytesForSmem>>>(d_origInput2,dxInt,threadHashes,threadReads,sobel_kern_1,sobel_kern_2,cols,rows,objToKernel);
        calcSobel_dX_k2_Precise<<<nblocks,threadsPerBlock,bytesForSmem>>>(dxInt,dxOutput,sobel_kern_1,sobel_kern_2,cols,rows,objToKernel);

        calcSobel_dY_k1_Precise<<<nblocks,threadsPerBlock,bytesForSmem>>>(d_origInput3,dyInt,sobel_kern_1,sobel_kern_2,cols,rows,objToKernel);
        calcSobel_dY_k2_Precise<<<nblocks,threadsPerBlock,bytesForSmem>>>(dyInt,dyOutput,sobel_kern_1,sobel_kern_2,cols,rows,objToKernel);

        /*
        calcSobel_dX_k1_Precise<<<nblocks,threadsPerBlock,bytesForSmem>>>(resizeOutput,dxInt_small,threadHashes,threadReads,sobel_kern_1,sobel_kern_2,resizedCols,resizedRows,objToKernel);
        calcSobel_dX_k2_Precise<<<nblocks,threadsPerBlock,bytesForSmem>>>(dxInt_small,dxOutput_small,sobel_kern_1,sobel_kern_2,resizedCols,resizedRows,objToKernel);

        calcSobel_dY_k1_Precise<<<nblocks,threadsPerBlock,bytesForSmem>>>(resizeOutput,dyInt_small,sobel_kern_1,sobel_kern_2,resizedCols,resizedRows,objToKernel);
        calcSobel_dY_k2_Precise<<<nblocks,threadsPerBlock,bytesForSmem>>>(dyInt_small,dyOutput_small,sobel_kern_1,sobel_kern_2,resizedCols,resizedRows,objToKernel);
        */
    }

    cudaDeviceSynchronize();
    if(train_set == true) { 
        // we are synched here, now we can print out the training set (if we are frame 0)
        HANDLE_ERROR( cudaMemcpy(reads,threadReads,5*rows*cols*sizeof(int),cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy(hashes,threadHashes,5*rows*cols*sizeof(int),cudaMemcpyDeviceToHost) );
        for(int i = 0;i < 5*rows*cols;i+=5) {
            /*
            if(!(reads[i] != reads[i]))
                printf("Global history hash [%d], value: %d\n",hashes[i],reads[i]);
            if(!(reads[i+1] != reads[i+1]))
                printf("Global history hash [%d], value: %d\n",hashes[i+1],reads[i+1]);
            if(!(reads[i+2] != reads[i+2]))
                printf("Global history hash [%d], value: %d\n",hashes[i+2],reads[i+2]);
            if(!(reads[i+3] != reads[i+3]))
                printf("Global history hash [%d], value: %d\n",hashes[i+3],reads[i+3]);
                */
            if(!(reads[i+4] != reads[i+4]))
                printf("Global history hash [%d], value: %d\n",hashes[i+4],reads[i+4]);
        }
        cudaFree(threadHashes);
        cudaFree(threadReads);
        free(reads);
        free(hashes);
    }

    // deep copy into the destination F2D structures
    //cout << "Creating image pyramid." << endl;
    ImagePyramid* retStruct = (ImagePyramid*)malloc(sizeof(ImagePyramid));
    // alloc these sub-arrays as pinned memory (required for copyAsync)
    retStruct->blurredImg = iSetArray(rows,cols,0);
    retStruct->resizedImg = iSetArray(resizedRows,resizedCols,0);
    retStruct->horizEdge = iSetArray(rows,cols,0);
    retStruct->vertEdge = iSetArray(rows,cols,0);
    retStruct->horizEdge_small = iSetArray(resizedRows,resizedCols,0);
    retStruct->vertEdge_small = iSetArray(resizedRows,resizedCols,0);

   
    cudaMemcpy((void*)&(retStruct->blurredImg->data[0]),d_outputPixels,rows*cols*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)&(retStruct->resizedImg->data[0]),resizeOutput,resizedRows*resizedCols*sizeof(int),cudaMemcpyDeviceToHost);
    //printf("addr of vertEdge->data: %p\n",retStruct->vertEdge->data);
    //printf("addr of horizEdge->data: %p\n",retStruct->horizEdge->data);
    cudaMemcpy((void*)retStruct->vertEdge->data,dxOutput,rows*cols*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)&(retStruct->horizEdge->data[0]),dyOutput,rows*cols*sizeof(int),cudaMemcpyDeviceToHost);   
    cudaMemcpy((void*)&(retStruct->vertEdge_small->data[0]),dxOutput_small,resizedRows*resizedCols*sizeof(int),cudaMemcpyDeviceToHost);   
    cudaMemcpy((void*)&(retStruct->horizEdge_small->data[0]),dyOutput_small,resizedRows*resizedCols*sizeof(int),cudaMemcpyDeviceToHost);   

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
    
    cudaFree(d_origInput);
    cudaFree(d_origInput2);
    cudaFree(d_origInput3);
    iFreeHandle(origPixelInput);

    return retStruct;
}

void destroyImgPyramid(ImagePyramid* retStruct, int imgNum )
{
    assert(retStruct != 0);
    //cout << "Destroying image pyramid for frame " << imgNum << endl;

    iFreeHandle(retStruct->blurredImg);
    iFreeHandle(retStruct->resizedImg);
    iFreeHandle(retStruct->horizEdge);
    iFreeHandle(retStruct->vertEdge);
    iFreeHandle(retStruct->horizEdge_small);
    iFreeHandle(retStruct->vertEdge_small);
    free(retStruct);
}

