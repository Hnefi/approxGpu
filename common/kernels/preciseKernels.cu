#include "preciseKernels.h"
#include "ghbFunctions.h"

#define SINGLEDIMINDEX(i,j,width) ((i)*(width) + (j))
#define SOBEL_RADIUS 1
#define BLUR_RADIUS 2
#define DIAMETER (2*BLUR_RADIUS + 1)


__global__ void calcSobel_dY_k1_Precise(float* inputPixels, float* intermediate, 
                                    int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t tref)
{
    int NUM_TEX_SCALED = 0;

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
                    for(int ii = -SOBEL_RADIUS;ii <= (SOBEL_RADIUS - NUM_TEX_SCALED);ii++) {
                        int location = curElement + SINGLEDIMINDEX(ii,0,width);
                        int filterWeightLoc = SOBEL_RADIUS + ii;
                        // bounds check #2 for surrounding pix
                        if (location < (width*height) && location >= 0) {
                            float loaded = inputPixels[location];
                            tmp += loaded * kernel_2[filterWeightLoc];
                            updateGHB(&(ghb[my_ghb_index]),loaded);
                        }
                    }

                    // finish up last few values with NUM_TEX reads
                    for(int ii = (SOBEL_RADIUS-NUM_TEX_SCALED+1); ii <= SOBEL_RADIUS;ii++) {
                        int filterWeightLoc = SOBEL_RADIUS + ii;
                        float curValueHash = hashGHB(&ghb[my_ghb_index]);
                        float texVal = tex1D<float>(tref,curValueHash);
                        tmp += (ghb[my_ghb_index+2] + texVal) * kernel_2[filterWeightLoc];
                    }
                    float avg = (float)tmp / kernelSum_2;
                    intermediate[curElement] = avg;
                }
            }
        }
    }
}

__global__ void calcSobel_dY_k2_Precise(float* intermediate, float* outputPixels, 
                                    int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t tref)
{
    // assign id's
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int NUM_TEX_SCALED = 0;

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
                    for(int ii = -SOBEL_RADIUS;ii <= (SOBEL_RADIUS - NUM_TEX_SCALED);ii++) {
                        int location = curElement + ii;
                        int filterWeightLoc = SOBEL_RADIUS + ii;
                        // bounds check #2 for surrounding pix
                        if (location < (width*height) && location >= 0) {
                            float loaded = intermediate[location];
                            tmp += loaded * kernel_1[filterWeightLoc];
                            updateGHB(&(ghb[my_ghb_index]),loaded);
                        }
                    }

                    // finish up last few values with NUM_TEX reads
                    for(int ii = (SOBEL_RADIUS-NUM_TEX_SCALED+1); ii <= SOBEL_RADIUS;ii++) {
                        int filterWeightLoc = SOBEL_RADIUS + ii;
                        float curValueHash = hashGHB(&ghb[my_ghb_index]);
                        float texVal = tex1D<float>(tref,curValueHash);
                        tmp += (ghb[my_ghb_index+2] + texVal) * kernel_1[filterWeightLoc];
                    }
                    float avg = (float)tmp / kernelSum_1;
                    outputPixels[curElement] = avg;
                }
            }
        }
    }
}



__global__ void calcSobel_dX_k1_Precise(float* inputPixels, float* intermediate,
                                float* hashes, float* threadReads,
                                int* kernel_1,int* kernel_2, uint width, uint height,
                                cudaTextureObject_t tref)
{
    int NUM_TEX_SCALED = 0;

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
                      for(int ii = -SOBEL_RADIUS;ii <= (SOBEL_RADIUS - NUM_TEX_SCALED);ii++) {
                          int location = curElement + ii;
                          int filterWeightLoc = SOBEL_RADIUS + ii;
                          // bounds check #2 for surrounding pix
                          if (location < (width*height) && location >= 0) {
                              float loaded = inputPixels[location];
                              //hashes[scaled+filterWeightLoc] = ghb[my_ghb_index+2] - ghb[my_ghb_index+1];
                              //threadReads[scaled+filterWeightLoc] = loaded - ghb[my_ghb_index+2];
                              tmp += loaded * kernel_2[filterWeightLoc];
                              updateGHB(&(ghb[my_ghb_index]),loaded);
                          }
                      }

                      // finish up last few values with NUM_TEX reads
                      for(int ii = (SOBEL_RADIUS-NUM_TEX_SCALED+1); ii <= SOBEL_RADIUS;ii++) {
                          int filterWeightLoc = SOBEL_RADIUS + ii;
                          float curValueHash = hashGHB(&ghb[my_ghb_index]);
                          float texVal = tex1D<float>(tref,curValueHash);
                          tmp += (ghb[my_ghb_index+2] + texVal) * kernel_2[filterWeightLoc];
                      }
                      float avg = (float)tmp / kernelSum_2;
                      intermediate[curElement] = avg;
                  }
            }
        }
    }
}

__global__ void calcSobel_dX_k2_Precise(float* intermediate, float* outputPixels, 
                                    int* kernel_1,int* kernel_2, uint width, uint height,cudaTextureObject_t tref)
{
    // assign id's
    int NUM_TEX= 0;
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
                    for(int ii = -SOBEL_RADIUS;ii <= (SOBEL_RADIUS - NUM_TEX);ii++) {
                        int location = curElement + SINGLEDIMINDEX(ii,0,width);
                        int filterWeightLoc = SOBEL_RADIUS + ii;
                        // bounds check #2 for surrounding pix
                        if (location < (width*height) && location >= 0) {
                            float loaded = intermediate[location];
                            tmp += loaded * kernel_1[filterWeightLoc];
                            updateGHB(&(ghb[my_ghb_index]),loaded);
                        }
                    }

                    // finish up last few values with NUM_TEX reads
                    for(int ii = (SOBEL_RADIUS-NUM_TEX+1); ii <= SOBEL_RADIUS;ii++) {
                        int filterWeightLoc = SOBEL_RADIUS + ii;
                        float curValueHash = hashGHB(&ghb[my_ghb_index]);
                        float texVal = tex1D<float>(tref,curValueHash);
                        tmp += (ghb[my_ghb_index+2] + texVal) * kernel_1[filterWeightLoc];
                    }
                    float avg = (float)tmp / kernelSum_1;
                    outputPixels[curElement] = avg;
                }
            }
        }
    }
}



__global__ void blurKernel_st1_Precise(int* inputPixels, float* intermediate, int* weightedKernel,float* hashes, float* threadReads,uint width, uint height,cudaTextureObject_t tref/*, other arguments */)
{
    int NUM_TEX = 0;
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

              if( jdx < height-2 && jdx > 1
                && idx < width-2 && idx > 1 ){
                //int curElement = (width * jdx) + idx;
                  int curElement = SINGLEDIMINDEX(jdx,idx,width);
                  int scaled = curElement * 5;

                  // a bit of manual loop unrolling - base the gmem read on the center
                  // pix, and then approximate all pixels around it from tex mem
                  int location = curElement;
                  int filterWeightLoc = BLUR_RADIUS;
                  if(location < (width*height) && location >=0) {

                      for(int ii = -BLUR_RADIUS;ii <= (BLUR_RADIUS - NUM_TEX);ii++) {
                          location = curElement + ii;
                          filterWeightLoc = BLUR_RADIUS + ii;
                          // bounds check #2 for surrounding pix
                          if (location < (width*height) && location >= 0) {
                              float loaded = inputPixels[location];
                              tmp += loaded * weightedKernel[filterWeightLoc];
                              updateGHB(&(ghb[my_ghb_index]),loaded);
                          }
                      }

                      // finish up last few values with NUM_TEX reads
                      for(int ii = (BLUR_RADIUS-NUM_TEX+1); ii <= BLUR_RADIUS;ii++) {
                          filterWeightLoc = BLUR_RADIUS + ii;
                          float curValueHash = hashGHB(&ghb[my_ghb_index]);
                          float texVal = tex1D<float>(tref,curValueHash);
                          tmp += (ghb[my_ghb_index+2] + texVal) * weightedKernel[filterWeightLoc];
                      }
                  }
                float avg = (float)tmp / kernelSum;
                intermediate[curElement] = avg;
            }
        }
    }
}


__global__ void blurKernel_st2_Precise(float* outputPixels,float* intermediate, int* weightedKernel,uint width, uint height,cudaTextureObject_t tref/*, other arguments */)
{
    int NUM_TEX = 0;
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
                  for(int ii = -BLUR_RADIUS;ii <= (BLUR_RADIUS - NUM_TEX);ii++) {
                      int location = curElement + (ii*width);
                      int filterWeightLoc = BLUR_RADIUS + ii;
                      // bounds check #2 for surrounding pix
                      if (location < (width*height) && location >= 0) {
                          float loaded = intermediate[location];
                          tmp += loaded * weightedKernel[filterWeightLoc];
                          updateGHB(&(ghb[my_ghb_index]),loaded);
                      }
                  }

                  // finish up last few values with NUM_TEX reads
                  for(int ii = (BLUR_RADIUS-NUM_TEX+1); ii <= BLUR_RADIUS;ii++) {
                      int filterWeightLoc = BLUR_RADIUS + ii;
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




__global__ void resizeKernel_st1_Precise(float* inputPixels,float* intermediate, int* weightedKernel,uint height, uint width,uint resizedRows,uint resizedCols,cudaTextureObject_t tref/*, other arguments */)
{
    int NUM_TEX = 0;
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

                for(int ii = 0;ii < (DIAMETER - NUM_TEX);ii++) {
                    int location = elemToReadFrom + ii;
                    // bounds check #2 for surrounding pix
                    if (location < (width*height) && location >= 0) {
                        float loaded = inputPixels[location];
                        tmp += loaded * weightedKernel[ii];
                        updateGHB(&(ghb[my_ghb_index]),loaded);
                    }
                }

                // finish up last few values with NUM_TEX reads
                for(int ii = (DIAMETER-NUM_TEX); ii < DIAMETER;ii++) {
                    float curValueHash = hashGHB(&ghb[my_ghb_index]);
                    float texVal = tex1D<float>(tref,curValueHash);
                    tmp += (ghb[my_ghb_index+2] + texVal) * weightedKernel[ii];
                }
                float avg = tmp / kernelSum;
                intermediate[elemToWrite] = avg;
            }
        }
    }
}



__global__ void resizeKernel_st2_Precise(float* outputPixels,float* intermediate, int* weightedKernel,uint height, uint width,uint resizedRows,uint resizedCols,cudaTextureObject_t tref/*, other arguments */)
{
    int NUM_TEX = 0;
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
                    tmp += (ghb[my_ghb_index+2] + texVal) * weightedKernel[ii];
                }
                float avg = tmp / kernelSum;
                outputPixels[elemToWrite] = avg;
            }
        }
    }
}
