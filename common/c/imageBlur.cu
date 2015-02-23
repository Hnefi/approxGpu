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

//F2D* imageBlur(I2D* imageIn)
TwoStepKernel* imageBlur(I2D* imageIn)
{
    int rows, cols;
    F2D *imageOut, *tempOut;
    float temp;
    I2D *kernel;
    int k, kernelSize, startCol, endCol, halfKernel, startRow, endRow, i, j, kernelSum;

    rows = imageIn->height;
    cols = imageIn->width;

    TwoStepKernel* ret = (TwoStepKernel*) malloc(sizeof(TwoStepKernel));
    ret->final = fSetArray(rows,cols,0);
    ret->intermediate = fSetArray(rows,cols,0);

    imageOut = fSetArray(rows, cols, 0);
    tempOut = fSetArray(rows, cols, 0);
    kernel = iMallocHandle(1, 5); // 1 row, 5 columns of kernel (NON SQUARE)

    asubsref(kernel,0) = 1; //kernel->data[0] = 1;
    asubsref(kernel,1) = 4;
    asubsref(kernel,2) = 6;
    asubsref(kernel,3) = 4;
    asubsref(kernel,4) = 1;
    kernelSize = 5;
    kernelSum = 16;

    startCol = 2;  
    endCol = cols - 2;  
    halfKernel = 2;   

    startRow = 2;    
    endRow = rows - 2;  
#ifdef APPROXIMATE
    // approximate all of the image data. NOTE: NOT THE KERNEL
    LVA_FUNCTION(2 /* int */, &(imageIn->data[0]),&(imageIn->data[rows*cols]),1);
    LVA_FUNCTION(5/*float*/, &(tempOut->data[0]),&(tempOut->data[rows*cols]),1);
    LVA_FUNCTION(5/*float*/, &(imageOut->data[0]),&(imageOut->data[rows*cols]),1);
#endif
    for(i=startRow; i<endRow; i++){
        for(j=startCol; j<endCol; j++)
        {
            temp = 0;
            for(k=-halfKernel; k<=halfKernel; k++)
            {
                temp += subsref(imageIn,i,j+k) * asubsref(kernel,k+halfKernel);
            }
            //subsref(tempOut,i,j) = temp/kernelSum;
            subsref(ret->intermediate,i,j) = temp/kernelSum;
        }
    }
    
    for(i=startRow; i<endRow; i++)
    {
        for(j=startCol; j<endCol; j++)
        {
            temp = 0;
            for(k=-halfKernel; k<=halfKernel; k++)
            {
                //temp += subsref(tempOut,(i+k),j) * asubsref(kernel,k+halfKernel);
                temp += subsref(ret->intermediate,(i+k),j) * asubsref(kernel,k+halfKernel);
            }
            //subsref(imageOut,i,j) = temp/kernelSum;
            subsref(ret->final,i,j) = temp/kernelSum;
        }
    }
#ifdef APPROXIMATE
    LVA_FUNCTION_RM(2 /* int */ ,&(imageIn->data[0]),&(imageIn->data[rows*cols]),1);
    LVA_FUNCTION_RM(5/*float*/ ,&(tempOut->data[0]),&(tempOut->data[rows*cols]),1);
    LVA_FUNCTION_RM(5/*float*/ ,&(imageOut->data[0]),&(imageOut->data[rows*cols]),1);
#endif
    fFreeHandle(tempOut);
    iFreeHandle(kernel);
    //return imageOut;
    return ret;
}
             

