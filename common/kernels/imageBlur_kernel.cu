// Mark Sutherland, Josh San Migue
//  - U of Toronto

// Global memory-based array image blur. non-optimized

#include "imageBlur_kernel.h"
#include "ghbFunctions.h"

#define RADIUS 2
#define SINGLEDIMINDEX(i,j,width) ((i)*(width) + (j))

__global__ void blurKernel_st1(int* inputPixels, int* intermediate, int* weightedKernel,int* hashes, int* threadReads,uint width, uint height,cudaTextureObject_t tref,int NUM_TEX/*, other arguments */)
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
    int kernelSum = 16;
    
    int ghb0 = 0; int ghb1 = 0; extern __shared__ int sharedghb[]; // for per-thread local history
    int my_ghb_index = ((threadIdx.y * blockDim.x) + threadIdx.x) * 3;

    // still check for this in case of small img, not all threads need execute
    for (int idx = i; idx < (i + xScale); idx++) {
        if ( ((idx == i+xScale) && xmod == 0) ||
             ((idx == i+xScale) && width <= totalX )) break; // exact mult. corner case
        for (int jdx = j; jdx < (j + yScale); jdx++) { // over each element to proc
            if( ((jdx == j + yScale) && ymod == 0) ||
                ((jdx == j + yScale) && height <= totalY)) break; // same corner case
            int tmp = 0;

              if( jdx < height-2 && jdx > 1
                && idx < width-2 && idx > 1 ){
                //int curElement = (width * jdx) + idx;
                  int curElement = SINGLEDIMINDEX(jdx,idx,width);
                  int scaled = curElement * 5;

                  // a bit of manual loop unrolling - base the gmem read on the center
                  // pix, and then approximate all pixels around it from tex mem
                  int location = curElement;
                  int filterWeightLoc = RADIUS;
                  if(location < (width*height) && location >=0) {

                      for(int ii = -RADIUS;ii <= (RADIUS - NUM_TEX);ii++) {
                          location = curElement + ii;
                          filterWeightLoc = RADIUS + ii;
                          // bounds check #2 for surrounding pix
                          if (location < (width*height) && location >= 0) {
                              int loaded = inputPixels[location];
#if 0 // training set generation
                              hashes[scaled + filterWeightLoc] = ghb1;
                              threadReads[scaled + filterWeightLoc] = loaded;
#endif
                              tmp += loaded * weightedKernel[filterWeightLoc];
                              ghb0 = ghb1; ghb1 = loaded;
                          }
                      }

                      // finish up last few values with NUM_TEX reads
                      for(int ii = (RADIUS-NUM_TEX+1); ii <= RADIUS;ii++) {
                          filterWeightLoc = RADIUS + ii;
                          int curValueHash = (ghb1 - ghb0) - NORM_MIN;
                          int texVal = tex1D<int>(tref,curValueHash);
                          tmp += (int)(ghb1 + texVal) * weightedKernel[filterWeightLoc];
                      }
                  }
                int avg = (int)tmp / kernelSum;
                intermediate[curElement] = avg;
            }
        }
    }
}
