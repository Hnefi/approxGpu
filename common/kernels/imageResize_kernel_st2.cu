// Mark Sutherland, Josh San Miguel
//  - U of Toronto

// Global memory-based array image resize. non-optimized

#include "imageResize_kernel_st2.h"
#include "ghbFunctions.h"

#define RADIUS 2
#define DIAMETER (2*RADIUS+1)
#define SINGLEDIMINDEX(i,j,width) ((i)*(width) + (j))

__global__ void resizeKernel_st2(float* outputPixels,float* intermediate, int* weightedKernel,uint height, uint width,uint resizedRows,uint resizedCols,cudaTextureObject_t tref /*, other arguments */)
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
    
    for (int idx = i; idx < (i + xScale); idx++) {
        if ( ((idx == i+xScale) && xmod == 0) ||
             ((idx == i+xScale) && width <= totalX )) break; // exact mult. corner case
        for (int jdx = j; jdx < (j + yScale); jdx++) { // over each element to proc
            if( ((jdx == j + yScale) && ymod == 0) ||
                ((jdx == j + yScale) && height <= totalY)) break; // same corner case*/
            float tmp = 0.0;

            if(idx >=0 && idx < resizedCols
            && jdx >= 0 && jdx < resizedRows-2 ) { // over all row/col
                int elemToWrite = SINGLEDIMINDEX(jdx,idx,resizedCols);
                int elemToReadFrom = SINGLEDIMINDEX((jdx*2),idx,resizedCols);

                for(int ii = 0;ii < (DIAMETER - NUM_TEX);ii++) {
                    int location = elemToReadFrom + SINGLEDIMINDEX(ii,0,resizedCols);
                    // bounds check #2 for surrounding pix
                    if (location < (width*height) && location >= 0) {
                        float loaded = intermediate[location];
                        tmp += loaded * weightedKernel[ii];
                        updateGHB(&(ghb[my_ghb_index]),loaded);
                    }
                }

                // finish up last few values with NUM_TEX reads
                for(int ii = (DIAMETER-NUM_TEX); ii < DIAMETER;ii++) {
                    float curValueHash = hashGHB(&ghb[my_ghb_index]);
                    float texVal = tex1D<float>(tref,curValueHash);
                    tmp += texVal * weightedKernel[ii];
                }
                float avg = tmp / kernelSum;
                outputPixels[elemToWrite] = avg;
            }
        }
    }
}
