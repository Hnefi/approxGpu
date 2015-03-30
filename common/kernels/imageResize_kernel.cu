// Mark Sutherland, Josh San Miguel
//  - U of Toronto

// Global memory-based array image resize. non-optimized

#include "imageResize_kernel.h"
#include "ghbFunctions.h"

#define RADIUS 2
#define DIAMETER (2*RADIUS + 1)
#define SINGLEDIMINDEX(i,j,width) ((i)*(width) + (j))

__global__ void resizeKernel_st1(int* inputPixels,int* intermediate, int* weightedKernel,uint height, uint width,uint resizedRows,uint resizedCols,cudaTextureObject_t tref,int NUM_TEX/*, other arguments */)
{
    // assign id's
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y; int totalX = gridDim.x * blockDim.x;
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

    int kernelSum = 16;

    int ghb0 = 0; int ghb1 = 0; extern __shared__ int sharedghb[]; // for per-thread local history
    int my_ghb_index = ((threadIdx.y * blockDim.x) + threadIdx.x) * 3;

    // still check for this in case of small img, not all threads need execute

    for (int idx = i; idx < (i + xScale); idx++) {
        if ( ((idx == i+xScale) && xmod == 0) ||
             ((idx == i+xScale) && width <= totalX )) break; // exact mult. corner case
        for (int jdx = j; jdx < (j + yScale); jdx++) { // over each element to proc
            
            if( ((jdx == j + yScale) && ymod == 0) ||
                ((jdx == j + yScale) && height <= totalY)) break; // same corner cas

            int tmp = 0;
            if(jdx < height-2 && jdx >= 2
            && idx < resizedCols-2 && idx >= 0) { // bounds check #1
                int elemToWrite = SINGLEDIMINDEX(jdx,idx,resizedCols);
                int elemToReadFrom = SINGLEDIMINDEX(jdx,(idx*2),width);

                for(int ii = 0;ii < (DIAMETER - NUM_TEX);ii++) {
                    int location = elemToReadFrom + ii;
                    // bounds check #2 for surrounding pix
                    if (location < (width*height) && location >= 0) {
                        int loaded = inputPixels[location];
                        tmp += loaded * weightedKernel[ii];
                        ghb0 = ghb1; ghb1 = loaded;
                    }
                }

                // finish up last few values with NUM_TEX reads
                for(int ii = (DIAMETER-NUM_TEX); ii < DIAMETER;ii++) {
                    int curValueHash = (ghb1 - ghb0) - NORM_MIN;
                    int texVal = (int)tex1D<float>(tref,(float)curValueHash);
                    tmp += (int)(ghb1 + texVal) * weightedKernel[ii];
                }
                int avg = tmp / kernelSum;
                intermediate[elemToWrite] = avg;
            }
        }
    }
}
