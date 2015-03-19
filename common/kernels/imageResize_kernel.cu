// Mark Sutherland, Josh San Miguel
//  - U of Toronto

// Global memory-based array image resize. non-optimized

#include "imageResize_kernel.h"
#include "ghbFunctions.h"
#include "texRef.h"

#define RADIUS 2
#define SINGLEDIMINDEX(i,j,width) ((i)*(width) + (j))

__global__ void resizeKernel_st1(float* inputPixels,float* intermediate, int* weightedKernel,uint height, uint width,uint resizedRows,uint resizedCols/*, other arguments */)
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

    float kernelSum = 16.0;

    extern __shared__ float ghb[]; // for per-thread local history
    int my_ghb_index = ((threadIdx.y * blockDim.x) + threadIdx.x) * 3;

    // still check for this in case of small img, not all threads need execute

    for (int idx = i; idx < (i + xScale); idx++) {
        if ( ((idx == i+xScale) && xmod == 0) ||
             ((idx == i+xScale) && width <= totalX )) break; // exact mult. corner case
        for (int jdx = j; jdx < (j + yScale); jdx++) { // over each element to proc
            
            if( ((jdx == j + yScale) && ymod == 0) ||
                ((jdx == j + yScale) && height <= totalY)) break; // same corner cas

            float tmp = 0.0;
            if(jdx < height-2 && jdx >= 2
            && idx < resizedCols-2 && idx >= 0) { // bounds check #1
                int elemToWrite = SINGLEDIMINDEX(jdx,idx,resizedCols);
                int elemToReadFrom = SINGLEDIMINDEX(jdx,(idx*2),width);

                // loop peel the first iteration for a gmem read
                // and then approx. the rest of the loads
                float loaded = inputPixels[elemToReadFrom];
                tmp += loaded * weightedKernel[0];
                updateGHB(&ghb[my_ghb_index],loaded);

                for (int ii = 1;ii <= 2*RADIUS;ii++) {
                    int location = elemToReadFrom + ii;
                    // bounds check #2 for surrounding pix
                    if (location < (width*height) && location >= 0) {
                        float curValueHash = hashGHB(&ghb[my_ghb_index]);
                        float texVal = tex1D(tref,curValueHash);
                        tmp += texVal * weightedKernel[ii];
                        updateGHB(&ghb[my_ghb_index],texVal);
                    }
                }
                float avg = tmp / kernelSum;
                intermediate[elemToWrite] = avg;
            }
        }
    }
}
