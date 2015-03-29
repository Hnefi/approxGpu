// Mark Sutherland, Josh San Miguel
//  - U of Toronto

// Global memory-based array sobel filtering - non optimized

#include "calcSobel_dY_kernel.h"
#include "ghbFunctions.h"

#define SINGLEDIMINDEX(i,j,width) ((i)*(width) + (j))
#define RADIUS 1

__global__ void calcSobel_dY_k1(int* inputPixels, int* intermediate, 
                                    int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t tref,int NUM_TEX)
{
    int NUM_TEX_SCALED = NUM_TEX;
    if(NUM_TEX > 2)
        NUM_TEX_SCALED = 2;

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
    int kernelSum_1 = 4;
    int kernelSum_2 = 2;
    int ghb0 = 0; int ghb1 = 0; extern __shared__ int sharedghb[]; // for per-thread local history
    int my_ghb_index = ((threadIdx.y * blockDim.x) + threadIdx.x) * 3;

    // still check for this in case of small img, not all threads need execute
    for (int idx = i; idx < (i + xScale); idx++) {
        if ( ((idx == i+xScale) && xmod == 0) ||
             ((idx == i+xScale) && width <= totalX )) break; // exact mult. corner case
        for (int jdx = j; jdx < (j + yScale); jdx++) { // over each element to proc
            if( ((jdx == j + yScale) && ymod == 0) ||
                ((jdx == j + yScale) && height <= totalY)) break; // same corner case

            if( idx > 0 && idx < width-1
                    && jdx > 0 && jdx < height-1) {
                int tmp = 0;
                int curElement = SINGLEDIMINDEX(jdx,idx,width);
                if(curElement < (width*height) && curElement >=0) {
                    for(int ii = -RADIUS;ii <= (RADIUS - NUM_TEX_SCALED);ii++) {
                        int location = curElement + SINGLEDIMINDEX(ii,0,width);
                        int filterWeightLoc = RADIUS + ii;
                        // bounds check #2 for surrounding pix
                        if (location < (width*height) && location >= 0) {
                            int loaded = inputPixels[location];
                            tmp += loaded * kernel_2[filterWeightLoc];
                            ghb0 = ghb1; ghb1 = loaded;
                        }
                    }

                    // finish up last few values with NUM_TEX reads
                    for(int ii = (RADIUS-NUM_TEX_SCALED+1); ii <= RADIUS;ii++) {
                        int filterWeightLoc = RADIUS + ii;
                        int curValueHash = (ghb1 - ghb0) - NORM_MIN;
                        int texVal = tex1D<int>(tref,curValueHash);
                        tmp += (int)(ghb1 + texVal) * kernel_2[filterWeightLoc];
                    }
                    int avg = (int)tmp / kernelSum_2;
                    intermediate[curElement] = avg;
                }
            }
        }
    }
}

__global__ void calcSobel_dY_k2(int* intermediate, int* outputPixels, 
                                    int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t tref,int NUM_TEX)
{
    // assign id's
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int NUM_TEX_SCALED = NUM_TEX;
    if(NUM_TEX > 2)
        NUM_TEX_SCALED = 2;

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
    int kernelSum_1 = 4;
    int kernelSum_2 = 2;
    int ghb0 = 0; int ghb1 = 0; extern __shared__ int sharedghb[]; // for per-thread local history
    int my_ghb_index = ((threadIdx.y * blockDim.x) + threadIdx.x) * 3;

    // still check for this in case of small img, not all threads need execute
    for (int idx = i; idx < (i + xScale); idx++) {
        if ( ((idx == i+xScale) && xmod == 0) ||
             ((idx == i+xScale) && width <= totalX )) break; // exact mult. corner case
        for (int jdx = j; jdx < (j + yScale); jdx++) { // over each element to proc
            if( ((jdx == j + yScale) && ymod == 0) ||
                ((jdx == j + yScale) && height <= totalY)) break; // same corner case

            if( idx > 0 && idx < width-1
                    && jdx > 0 && jdx < height-1) {
                int tmp = 0;
                int curElement = SINGLEDIMINDEX(jdx,idx,width);
                if(curElement < (width*height) && curElement >=0) {
                    for(int ii = -RADIUS;ii <= (RADIUS - NUM_TEX_SCALED);ii++) {
                        int location = curElement + ii;
                        int filterWeightLoc = RADIUS + ii;
                        // bounds check #2 for surrounding pix
                        if (location < (width*height) && location >= 0) {
                            int loaded = intermediate[location];
                            tmp += loaded * kernel_1[filterWeightLoc];
                            ghb0 = ghb1; ghb1 = loaded;
                        }
                    }

                    // finish up last few values with NUM_TEX reads
                    for(int ii = (RADIUS-NUM_TEX_SCALED+1); ii <= RADIUS;ii++) {
                        int filterWeightLoc = RADIUS + ii;
                        int curValueHash = (ghb1 - ghb0) - NORM_MIN;
                        int texVal = tex1D<int>(tref,curValueHash);
                        tmp += (int)(ghb1 + texVal) * kernel_1[filterWeightLoc];
                    }
                    int avg = (int)tmp / kernelSum_1;
                    outputPixels[curElement] = avg;
                }
            }
        }
    }
}
