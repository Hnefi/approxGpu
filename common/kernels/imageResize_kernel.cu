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
    int xScale = resizedCols / totalX;
    if (resizedCols % totalX)
        xScale += 1;
    int yScale = height / totalY;
    if (height % totalY)
        yScale += 1;
    int xmod = resizedCols % totalX;
    int ymod = height % totalY;

    if (xScale > 1) { // each thread sweep more than 1 elem in x direction
       i *= xScale;
    }

    if (yScale > 1) { // same thing in y dimension
        j *= yScale;
    }
    float kernelSum = 16.0;

    // still check for this in case of small img, not all threads need execute
    for (int idx = i; idx <= (i + xScale); idx++) {
        if ( ((idx == i+xScale) && xmod == 0) ||
             ((idx == i+xScale) && width <= totalX )) break; // exact mult. corner case
        for (int jdx = j; jdx <= (j + yScale); jdx++) { // over each element to proc
            
            if( ((jdx == j + yScale) && ymod == 0) ||
                ((jdx == j + yScale) && height <= totalY)) break; // same corner case
            float tmp = 0.0;

            if(jdx < height-1 && jdx > 1
            && idx < resizedCols && idx >= 0) { // bounds check #1
            //if(idx < width-2 && jdx < height-2
               //&& idx > 1 && jdx > 1) { 
                int elemToWrite = SINGLEDIMINDEX(jdx,idx,width);
                int elemToReadFrom = SINGLEDIMINDEX(jdx,(idx*2),width);
                //int curElement = (width * jdx) + idx;

                for (int ii = 0;ii <= 2*RADIUS;ii++) {
                    int location = elemToReadFrom + ii;
                    // bounds check #2 for surrounding pix
                    if (location < (width*height) && location >= 0) {
                        tmp += inputPixels[location]*weightedKernel[ii];
                        /*if (jdx == 2 && idx == 0) {
                            printf("I'm adding value %0.4f to the sum...\n",inputPixels[location]*weightedKernel[ii]);
                        }*/
                    }
                }
                float avg = tmp / kernelSum;
                /*if (jdx == 2 && idx == 0) {
                    printf("Final intermediate value is %0.4f\n.",avg);
                }*/
                intermediate[elemToWrite] = avg;
            }
        }
    }

    __syncthreads();
    /*if(i==0 && j==0) {
        for(int q = 2*width;q<=(2*width)+5;q++) {
            printf("\tElement # %d of GPU intermediate array is: %0.4f\n",q,intermediate[q]);
        }
    }*/

    // Re-divide work for "column aggregation", which elem to start on.
    // Only y-scale of threads changes, we still process the same
    //  limited number of columns in x-scale.
    yScale = resizedRows / totalY;
    if (resizedRows % totalY)
        yScale += 1;
    ymod = resizedRows % totalY;

    if (yScale > 1) {
        j *= yScale;
    }

    // still check for this in case of small img, not all threads need execute
    for (int idx = i; idx <= (i + xScale); idx++) {
        if ( ((idx == i+xScale) && xmod == 0) ||
             ((idx == i+xScale) && width <= totalX )) break; // exact mult. corner case
        for (int jdx = j; jdx <= (j + yScale); jdx++) { // over each element to proc
            if( ((jdx == j + yScale) && ymod == 0) ||
                ((jdx == j + yScale) && height <= totalY)) break; // same corner case
            float tmp = 0.0;

            if(idx >=0 && idx < resizedCols
            && jdx >= 0 && jdx < resizedRows) { // over all row/col
            //if(idx < width-2 && jdx < height-2
               //&& idx > 1 && jdx > 1) { // bounds check #1
                int elemToWrite = SINGLEDIMINDEX(jdx,idx,resizedCols);
                int elemToReadFrom = SINGLEDIMINDEX((jdx*2),idx,width);
                //int curElement = (width * jdx) + idx;

                for (int ii = 0;ii <= 2*RADIUS;ii++) {
                    int location = elemToReadFrom + (ii*width);
                    // bounds check #2 for surrounding pix
                    if (location < (width*height) && location >= 0) {
                        tmp += intermediate[location]*weightedKernel[ii];
                    }
                }
                float avg = tmp / kernelSum;
                outputPixels[elemToWrite] = avg;
            }
        }
    }
}
