#define RADIUS 2
#define DIAMETER (RADIUS*2) + 1
#define S DIAMETER*DIAMETER // size
#define CENTER S/2

#define SINGLEDIMINDEX(i,j,width) ((j)*(width) + (i))

__global__ void weightedBlurKernel(int* inputPixels, float* outputPixels,float* intermediate, int* weightedKernel,uint width, uint height /*, other arguments */)
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
    if (xScale >= 1) { // each thread sweep more than 1 elem in x direction
       i *= xScale;
    }

    if (yScale >= 1) { // same thing in y dimension
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

            if(idx < width-2 && jdx < height-2
               && idx > 1 && jdx > 1) { // bounds check #1
                int curElement = (width * jdx) + idx;

                for (int ii = -RADIUS;ii <= RADIUS;ii++) {
                    int location = curElement + ii;
                    int filterWeightLoc = RADIUS + ii;
                    // bounds check #2 for surrounding pix
                    if (location < (width*height) && location >= 0) {
                        tmp += inputPixels[location]*weightedKernel[filterWeightLoc];
                    }
                }
                float avg = (float)tmp / kernelSum;
                intermediate[curElement] = avg;
            }
        }
    }

    __syncthreads();

    // still check for this in case of small img, not all threads need execute
    for (int idx = i; idx <= (i + xScale); idx++) {
        if ( ((idx == i+xScale) && xmod == 0) ||
             ((idx == i+xScale) && width <= totalX )) break; // exact mult. corner case
        for (int jdx = j; jdx <= (j + yScale); jdx++) { // over each element to proc
            if( ((jdx == j + yScale) && ymod == 0) ||
                ((jdx == j + yScale) && height <= totalY)) break; // same corner case
            float tmp = 0.0;

            if(idx < width-2 && jdx < height-2
               && idx > 1 && jdx > 1) { // bounds check #1
                int curElement = (width * jdx) + idx;

                for (int ii = -RADIUS;ii <= RADIUS;ii++) {
                    int location = curElement + (ii*width);
                    int filterWeightLoc = RADIUS + ii;
                    // bounds check #2 for surrounding pix
                    if (location < (width*height) && location >= 0) {
                        tmp += intermediate[location]*weightedKernel[filterWeightLoc];
                    }
                }
                float avg = tmp / kernelSum;
                outputPixels[curElement] = avg;
            }
        }
    }
}
