// Mark Sutherland, Josh San Miguel
//  - U of Toronto

// Global memory-based array sobel filtering - non optimized

#include "calcSobel_dY_kernel.h"

#define RADIUS 1
#define SINGLEDIMINDEX(i,j,width) ((i)*(width) + (j))

__global__ void calcSobel_dY_k1(float* inputPixels, float* intermediate, 
                                    int* kernel_1,int* kernel_2, uint width, uint height)
{
    // assign id's
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int totalX = gridDim.x * blockDim.x;
    int totalY = gridDim.y * blockDim.y;

    /* Used to calculate the "ranges" each thread must span based on img size. */
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
    float kernelSum_1 = 4.0;
    float kernelSum_2 = 2.0;

    // still check for this in case of small img, not all threads need execute
    for (int idx = i; idx < (i + xScale); idx++) {
        if ( ((idx == i+xScale) && xmod == 0) ||
             ((idx == i+xScale) && width <= totalX )) break; // exact mult. corner case
        for (int jdx = j; jdx < (j + yScale); jdx++) { // over each element to proc
            if( ((jdx == j + yScale) && ymod == 0) ||
                ((jdx == j + yScale) && height <= totalY)) break; // same corner case

              if( idx > 0 && idx < width-1
               && jdx > 0 && jdx < height-1) {
                float tmp = 0.0;
                int curElement = SINGLEDIMINDEX(jdx,idx,width);

                for (int ii = -RADIUS;ii <= RADIUS;ii++) {
                    int location = curElement + SINGLEDIMINDEX(ii,0,width);
                    int filterWeightLoc = RADIUS + ii;
                    // bounds check #2 for surrounding pix
                    if (location < (width*height) && location >= 0) {
                        tmp += inputPixels[location]*kernel_2[filterWeightLoc];
                    }
                }
                float avg = (float)tmp / kernelSum_2;
                intermediate[curElement] = avg;
            }
        }
    }
}

__global__ void calcSobel_dY_k2(float* intermediate, float* outputPixels, 
                                    int* kernel_1,int* kernel_2, uint width, uint height)
{
    // assign id's
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int totalX = gridDim.x * blockDim.x;
    int totalY = gridDim.y * blockDim.y;

    /* Used to calculate the "ranges" each thread must span based on img size. */
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
    float kernelSum_1 = 4;
    float kernelSum_2 = 2;

    // still check for this in case of small img, not all threads need execute
    for (int idx = i; idx < (i + xScale); idx++) {
        if ( ((idx == i+xScale) && xmod == 0) ||
             ((idx == i+xScale) && width <= totalX )) break; // exact mult. corner case
        for (int jdx = j; jdx < (j + yScale); jdx++) { // over each element to proc
            if( ((jdx == j + yScale) && ymod == 0) ||
                ((jdx == j + yScale) && height <= totalY)) break; // same corner case

              if( idx > 0 && idx < width-1
               && jdx > 0 && jdx < height-1) {
                float tmp = 0.0;
                int curElement = SINGLEDIMINDEX(jdx,idx,width);

                for (int ii = -RADIUS;ii <= RADIUS;ii++) {
                    int location = curElement + ii;
                    int filterWeightLoc = RADIUS + ii;
                    // bounds check #2 for surrounding pix
                    if (location < (width*height) && location >= 0) {
                        tmp += intermediate[location]*kernel_1[filterWeightLoc];
                    }
                }
                float avg = (float)tmp / kernelSum_1;
                outputPixels[curElement] = avg;
            }
        }
    }
}
