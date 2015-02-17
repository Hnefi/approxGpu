// Mark Sutherland, Josh San Miguel
//  - U of Toronto

// Global memory-based array image resize. non-optimized

#include "imageResize_kernel.h"
#include <stdio.h>

#define RADIUS 2
#define SINGLEDIMINDEX(i,j,width) ((i)*(width) + (j))

__global__ void imageResizeKernel(float* inputPixels, float* outputPixels,float* intermediate, int* weightedKernel,uint height, uint width,uint resizedRows,uint resizedCols/*, other arguments */)
{
    // assign id's
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int totalX = gridDim.x * blockDim.x;
    int totalY = gridDim.y * blockDim.y;

    // Divide work for "row aggregation", which elem to start on
   
    int xScale = width / totalX;
    if (width % totalX)
        xScale += 1;
    int yScale = height / totalY;
    if (height % totalY)
        yScale += 1;
    int xmod = width % totalX;
    int ymod = height % totalY;

    if (xScale > 1) { // each thread sweep more than 1 elem in x direction
       i *= xScale;
    }

    if (yScale > 1) { // same thing in y dimension
        j *= yScale;
    }

    if ( i == 0 && j == 0)
        printf("width: %d, height: %d, resizedRows: %d, resizedCols: %d\nxScale: %d, yScale: %d, xmod: %d, ymod %d\n",
                width,height,resizedRows,resizedCols,xScale,yScale,xmod,ymod);
    
    float kernelSum = 16.0;

    // still check for this in case of small img, not all threads need execute
    
    for (int idx = i; idx < (i + xScale); idx++) {
        if ( ((idx == i+xScale) && xmod == 0) ||
             ((idx == i+xScale) && width <= totalX )) break; // exact mult. corner case
        for (int jdx = j; jdx < (j + yScale); jdx++) { // over each element to proc
            
            if( ((jdx == j + yScale) && ymod == 0) ||
                ((jdx == j + yScale) && height <= totalY)) break; // same corner cas

            float tmp = 0.0;
            if(jdx < height-2 && jdx >= 0
            && idx < resizedCols-2 && idx >= 0) { // bounds check #1
            //if(i < width-2 && j < height-2
               //&& i > 1 && j > 1) { 
                int elemToWrite = SINGLEDIMINDEX(jdx,idx,resizedCols);
                int elemToReadFrom = SINGLEDIMINDEX(jdx,(idx*2),width);
                if (jdx == 30 && idx == 0) {
                    printf("elemToWrite: %d, elemToReadFrom: %d\n",elemToWrite,elemToReadFrom);
                }
                //int curElement = (width * j) + i;

                for (int ii = 0;ii <= 2*RADIUS;ii++) {
                    int location = elemToReadFrom + ii;
                    // bounds check #2 for surrounding pix
                    if (location < (width*height) && location >= 0) {
                        tmp += inputPixels[location]*weightedKernel[ii];
                        if (jdx == 30 && idx == 0) {
                            printf("read value %0.8f\n",inputPixels[location]);
                        }
                    }
                }
                float avg = tmp / kernelSum;
                  if (jdx == 30 && idx == 0) {
                    printf("Final intermediate value is %0.4f\n.",avg);
                }
                intermediate[elemToWrite] = avg;
            }
         
        }
    }

    __syncthreads();

    // Re-divide work for "column aggregation", which elem to start on.

    // Only y-scale of threads changes, we still process the same
    //  limited number of columns in x-scale.
    /*
    yScale = resizedRows / totalY;
    if (resizedRows % totalY)
        yScale += 1;
    ymod = resizedRows % totalY;

    if (yScale > 1) {
        j *= yScale;
    }*/

    // still check for this in case of small img, not all threads need execute
    /*
    for (int idx = i; idx <= (i + xScale); idx++) {
        if ( ((idx == i+xScale) && xmod == 0) ||
             ((idx == i+xScale) && width <= totalX )) break; // exact mult. corner case
        for (int jdx = j; jdx <= (j + yScale); jdx++) { // over each element to proc
            if( ((jdx == j + yScale) && ymod == 0) ||
                ((jdx == j + yScale) && height <= totalY)) break; // same corner case*/
            float tmp = 0.0;

            if(i >=0 && i < resizedCols-2
            && j >= 0 && j < resizedRows-2) { // over all row/col
            //if(idx < width-2 && jdx < height-2
               //&& idx > 1 && jdx > 1) { // bounds check #1
                int elemToWrite = SINGLEDIMINDEX(j,i,resizedCols);
                int elemToReadFrom = SINGLEDIMINDEX((j*2),i,resizedCols);
                //int curElement = (width * jdx) + idx;

                for (int ii = 0;ii <= 2*RADIUS;ii++) {
                    int location = elemToReadFrom + (ii*resizedCols);
                    // bounds check #2 for surrounding pix
                    if (location < (resizedCols*height) && location >= 0) {
                        tmp += intermediate[location]*weightedKernel[ii];
                    }
                }
                float avg = tmp / kernelSum;
                outputPixels[elemToWrite] = avg;
            }
            /*
        }
    }*/
}
