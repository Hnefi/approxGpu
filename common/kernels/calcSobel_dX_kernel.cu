// Mark Sutherland, Josh San Miguel
//  - U of Toronto

// Global memory-based array sobel filtering - non optimized

#include "calcSobel_dX_kernel.h"
#include "ghbFunctions.h"

#define RADIUS 1
#define SINGLEDIMINDEX(i,j,width) ((i)*(width) + (j))

//because these kernels are smaller
#if(NUM_TEX > 2)
#define NUM_TEX_SCALED 2
#else
#define NUM_TEX_SCALED (NUM_TEX)
#endif

__global__ void calcSobel_dX_k1(float* inputPixels, float* intermediate,
                                float* hashes, float* threadReads,
                                int* kernel_1,int* kernel_2, uint width, uint height,
                                cudaTextureObject_t tref)
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

    extern __shared__ float ghb[]; // for per-thread local history
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
                  float tmp = 0.0;
                  int curElement = SINGLEDIMINDEX(jdx,idx,width);
                  int scaled = curElement * 3; // because radius is 1
                  if(curElement < (width*height) && curElement >=0) {
                      for(int ii = -RADIUS;ii <= (RADIUS - NUM_TEX_SCALED);ii++) {
                          int location = curElement + ii;
                          int filterWeightLoc = RADIUS + ii;
                          // bounds check #2 for surrounding pix
                          if (location < (width*height) && location >= 0) {
                              float loaded = inputPixels[location];
                              hashes[scaled+filterWeightLoc] = ghb[my_ghb_index+2] - ghb[my_ghb_index+1];
                              threadReads[scaled+filterWeightLoc] = loaded - ghb[my_ghb_index+2];
                              tmp += loaded * kernel_2[filterWeightLoc];
                              updateGHB(&(ghb[my_ghb_index]),loaded);
                          }
                      }

                      // finish up last few values with NUM_TEX reads
                      for(int ii = (RADIUS-NUM_TEX_SCALED+1); ii <= RADIUS;ii++) {
                          int filterWeightLoc = RADIUS + ii;
                          float curValueHash = hashGHB(&ghb[my_ghb_index]);
                          float texVal = tex1D<float>(tref,curValueHash);
                          tmp += texVal * kernel_2[filterWeightLoc];
                      }
                      float avg = (float)tmp / kernelSum_2;
                      intermediate[curElement] = avg;
                  }
            }
        }
    }
}

__global__ void calcSobel_dX_k2(float* intermediate, float* outputPixels, 
                                    int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t tref)
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
    extern __shared__ float ghb[]; // for per-thread local history
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
                float tmp = 0.0;
                int curElement = SINGLEDIMINDEX(jdx,idx,width);
                if(curElement < (width*height) && curElement >=0) {
                    for(int ii = -RADIUS;ii <= (RADIUS - NUM_TEX);ii++) {
                        int location = curElement + SINGLEDIMINDEX(ii,0,width);
                        int filterWeightLoc = RADIUS + ii;
                        // bounds check #2 for surrounding pix
                        if (location < (width*height) && location >= 0) {
                            float loaded = intermediate[location];
                            tmp += loaded * kernel_1[filterWeightLoc];
                            updateGHB(&(ghb[my_ghb_index]),loaded);
                        }
                    }

                    // finish up last few values with NUM_TEX reads
                    for(int ii = (RADIUS-NUM_TEX+1); ii <= RADIUS;ii++) {
                        int filterWeightLoc = RADIUS + ii;
                        float curValueHash = hashGHB(&ghb[my_ghb_index]);
                        float texVal = tex1D<float>(tref,curValueHash);
                        tmp += texVal * kernel_1[filterWeightLoc];
                    }
                    float avg = (float)tmp / kernelSum_1;
                    outputPixels[curElement] = avg;
                }
            }
        }
    }
}
