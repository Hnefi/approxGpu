#include <stdio.h>
#include <stdlib.h>

#define SINGLEDIMINDEX(i,j,width) ((i)*(width) + (j))
//#define TEXMEM

// wrapper that prints error stuff
inline static void HandleError( cudaError_t err,
        const char *file,
        int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

texture<float,cudaTextureType2D,cudaReadModeElementType> tref;

// benchmarking kernel - use this to launch a bunch of memory reads & writes to see what happens
//      Launches 4 tex reference reads and 1 global memory write. 
__global__ void texMark(float* output, uint width, uint height )
{
    // assign id's
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int jdx = blockIdx.y * blockDim.y + threadIdx.y;

    float tmp = 0.0;

    if( jdx < height-2 && jdx > 1
            && idx < width-2 && idx > 1 ){
        int curElement = SINGLEDIMINDEX(jdx,idx,width);
        if(curElement < (width*height) && curElement >= 0) {
            tmp += ( tex2D(tref,idx+1,jdx) + tex2D(tref,idx-1,jdx) + tex2D(tref,idx,jdx+1) + tex2D(tref,idx,jdx-1) );
            tmp /= 4.0;
            output[curElement] = tmp;
        }
    }
}

__global__ void avgMark(float* input, float* output, uint width, uint height )
{
    // assign id's
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int jdx = blockIdx.y * blockDim.y + threadIdx.y;

    float tmp = 0.0;

    if( jdx < height-2 && jdx > 1
            && idx < width-2 && idx > 1 ){
        int curElement = SINGLEDIMINDEX(jdx,idx,width);
        int singleRowOffset = SINGLEDIMINDEX(1,0,width);
        if(curElement < (width*height) && curElement >= 0) {
            tmp += ( input[curElement-1] + input[curElement+1] + input[curElement + singleRowOffset] + input[curElement - singleRowOffset] );
            tmp /= 4.0;
            output[curElement] = tmp;
        }
    }
}

// test for texture memory
int main(int argc, char* argv[])
{
    unsigned int cols = 1280;
    unsigned int rows = 720;

    cudaDeviceReset();
    cudaDeviceProp dev_props;
    // assume device 0
    HANDLE_ERROR( cudaGetDeviceProperties(&dev_props,0) );

    // print some stuff
    printf("Current Device compute capability: %d.%d\n",dev_props.major,dev_props.minor);
    printf("2D texture memory space limits: %d x %d\n",dev_props.maxTexture2D[0],dev_props.maxTexture2D[1]);
    printf("2D texture memory (gathered) space limits: %d x %d\n",dev_props.maxTexture2DGather[0],dev_props.maxTexture2DGather[1]);

    // create a big random array
    float* big_arr = (float*)malloc(cols*rows*sizeof(float)); 
    for(int i = 0;i<rows*cols;i++)
        big_arr[i] = ((float) rand()) / ((float) RAND_MAX);

    float* out_arr = (float*)calloc(cols*rows,sizeof(float));
    // alloc output
    float* gmemOutput;
    HANDLE_ERROR( cudaMalloc((void**)&gmemOutput,rows*cols*sizeof(float)) );

    // set up the thread block sizes
    int rowsIn = floor((rows+1));
    int colsIn = floor((cols+1));
    int nBlocksWide = colsIn/32;
    if (colsIn % 32) nBlocksWide++;
    int nBlocksTall = rowsIn/32;
    if (rowsIn % 32) nBlocksTall++;
    dim3 nblocks(nBlocksWide,nBlocksTall);
    dim3 threadsPerBlock(32,32);
    cudaEvent_t kstart,kstop;
    HANDLE_ERROR( cudaEventCreate(&kstart) );
    HANDLE_ERROR( cudaEventCreate(&kstop) );
#ifdef TEXMEM
    // set up global memory and texture reference
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
    cudaArray* my_arr;
    HANDLE_ERROR( cudaMallocArray(&my_arr,&channelDesc,cols,rows) );

    // copy in
    HANDLE_ERROR( cudaMemcpyToArray(my_arr,0,0,big_arr,rows*cols*sizeof(float),cudaMemcpyHostToDevice) );

    // setup texture reference parameters
    tref.addressMode[0] = cudaAddressModeWrap; 
    tref.addressMode[1] = cudaAddressModeWrap; 
    tref.filterMode = cudaFilterModeLinear; 
    tref.normalized = false;
    HANDLE_ERROR( cudaBindTextureToArray(tref,my_arr,channelDesc) );
    // compute
    cudaEventRecord(kstart);
    texMark<<<nblocks,threadsPerBlock>>>(gmemOutput,cols,rows);
    texMark<<<nblocks,threadsPerBlock>>>(gmemOutput,cols,rows);
    texMark<<<nblocks,threadsPerBlock>>>(gmemOutput,cols,rows);
    texMark<<<nblocks,threadsPerBlock>>>(gmemOutput,cols,rows);
    HANDLE_ERROR( cudaPeekAtLastError() );
    cudaEventRecord(kstop);

    // copy out
    HANDLE_ERROR( cudaMemcpyFromArray(out_arr,my_arr,0,0,rows*cols*sizeof(float),cudaMemcpyDeviceToHost) );
#else
    float* gmemInput;
    HANDLE_ERROR( cudaMalloc((void**)&gmemInput,rows*cols*sizeof(float)) );

    // copy in
    HANDLE_ERROR( cudaMemcpy(gmemInput,big_arr,rows*cols*sizeof(float),cudaMemcpyHostToDevice) );

    // compute
    cudaEventRecord(kstart);
    avgMark<<<nblocks,threadsPerBlock>>>(gmemInput,gmemOutput,cols,rows);
    avgMark<<<nblocks,threadsPerBlock>>>(gmemInput,gmemOutput,cols,rows);
    avgMark<<<nblocks,threadsPerBlock>>>(gmemInput,gmemOutput,cols,rows);
    avgMark<<<nblocks,threadsPerBlock>>>(gmemInput,gmemOutput,cols,rows);
    HANDLE_ERROR( cudaPeekAtLastError() );
    cudaEventRecord(kstop);

    // copy out
    HANDLE_ERROR( cudaMemcpy(out_arr,gmemOutput,rows*cols*sizeof(float),cudaMemcpyDeviceToHost) );
#endif
    float time;
    cudaEventElapsedTime(&time,kstart,kstop);

    /*
    printf("----- AFTER -----\n");
    for(int i = 175;i<200;i++)
        printf("out_arr[i] = %0.5f\n",out_arr[i]);
    */

    printf("Execution time: %0.8f ms\n",time);

#ifdef TEXMEM
    cudaFreeArray(my_arr);
#else
    cudaFree(gmemInput);
#endif
    cudaFree(gmemOutput);
    free(big_arr);        
    free(out_arr);
    return 0;
}

void gmemTest(void)
{

}

/*
int main(char argc, char* argv[])
{
}
*/
