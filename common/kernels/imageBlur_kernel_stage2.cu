// Mark Sutherland, Josh San Miguel
//  - U of Toronto

// Global memory-based array image blur. non-optimized

#include "imageBlur_kernel_stage2.h"
#include "ghbFunctions.h"
#include <stdio.h>

#define RADIUS 2
#define SINGLEDIMINDEX(i,j,width) ((i)*(width) + (j))

__global__ void blurKernel_st2(float* outputPixels,float* intermediate, int* weightedKernel,uint width, uint height,cudaTextureObject_t tref,int NUM_TEX /*, other arguments */)
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
    float kernelSum = 16.0;
    extern __shared__ float ghb[]; // for per-thread local history
    int my_ghb_index = ((threadIdx.y * blockDim.x) + threadIdx.x) * 3;

    // still check for this in case of small img, not all threads need execute
    for (int idx = i; idx < (i + xScale); idx++) {
        if ( ((idx == i+xScale) && xmod == 0) ||
             ((idx == i+xScale) && width <= totalX )) break; // exact mult. corner case
        for (int jdx = j; jdx < (j + yScale); jdx++) { // over each element to proc
            if( ((jdx == j + yScale) && ymod == 0) ||
                ((jdx == j + yScale) && height <= totalY)) break; // same corner case
            float tmp = 0.0;

            if(idx < width-2 && idx > 1
               && jdx < height-2 && jdx > 1) { // bounds check #1
                  int curElement = SINGLEDIMINDEX(jdx,idx,width);
                  for(int ii = -RADIUS;ii <= (RADIUS - NUM_TEX);ii++) {
                      int location = curElement + (ii*width);
                      int filterWeightLoc = RADIUS + ii;
                      // bounds check #2 for surrounding pix
                      if (location < (width*height) && location >= 0) {
                          float loaded = intermediate[location];
                          tmp += loaded * weightedKernel[filterWeightLoc];
                          updateGHB(&(ghb[my_ghb_index]),loaded);
                      }
                  }

                  // finish up last few values with NUM_TEX reads
                  for(int ii = (RADIUS-NUM_TEX+1); ii <= RADIUS;ii++) {
                      int filterWeightLoc = RADIUS + ii;
                      float curValueHash = hashGHB(&ghb[my_ghb_index]);
                      float texVal = tex1D<float>(tref,curValueHash);
                      tmp += (ghb[my_ghb_index+2] + texVal) * weightedKernel[filterWeightLoc];
                  }
                float avg = tmp / kernelSum;
                outputPixels[curElement] = avg;
            }
        }
    }
}
