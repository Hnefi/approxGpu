#include <stdio.h>
#include <stdlib.h>
#include "genBuckets.h"

#define SINGLEDIMINDEX(i,j,width) ((i)*(width) + (j))
#define TEXMEM

// wrapper that prints error stuff
inline static void HandleError( cudaError_t err, const char *file,int line, bool quit = true ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
        if(quit) {
            exit(err);
        }
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

texture<float,cudaTextureType1D,cudaReadModeElementType> tref;

__device__ void updateGHB(float* mem_arr,float new_val)
{
    mem_arr[0] = mem_arr[1];
    mem_arr[1] = mem_arr[2];
    mem_arr[2] = new_val;
}

__device__ float hashGHB(float* mem_arr)
{
    float sum = mem_arr[0] + mem_arr[1] + mem_arr[2];
    sum /= 3.0;
    /*
       if (threadIdx.x == 0 && threadIdx.y == 0)
       printf("returning hash of %0.5f\n",sum);
     */
    return sum;
}

// benchmarking kernel - use this to launch a bunch of memory reads & writes to see what happens
//      Launches 4 tex reference reads and 1 global memory write. 
__global__ void texMark(float* input, float* output, uint width, uint height,int shift )
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

    extern __shared__ float ghb[];
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
                int curElement = SINGLEDIMINDEX(jdx,idx,width);
                //int singleRowOffset = SINGLEDIMINDEX(1,0,width);
                if(curElement < (width*height) && curElement >= 0) {
                    float loaded = input[curElement-1+shift];
                    tmp += loaded;
                    updateGHB(&ghb[my_ghb_index],loaded);

                    float curValueHash = hashGHB(&ghb[my_ghb_index]);
                    float texVal = tex1D(tref,curValueHash / 175.0);
                    tmp += texVal;
                    updateGHB(&ghb[my_ghb_index],texVal);

                    curValueHash = hashGHB(&ghb[my_ghb_index]);
                    texVal = tex1D(tref,curValueHash / 175.0);
                    tmp += texVal;
                    updateGHB(&ghb[my_ghb_index],texVal);
                    /*
                       float curValueHash = hashGHB(&(ghb[my_ghb_index])); // TODO: this value is 0 always right now (all threads
                    // will follow the same "path").
                    float texVal = tex1D(tref,curValueHash);
                    //updateGHB(&ghb[my_ghb_index],texVal);
                    tmp += texVal;

                    curValueHash = hashGHB(&ghb[my_ghb_index]);
                    texVal = tex1D(tref,curValueHash);
                    //updateGHB(&ghb[my_ghb_index],texVal);
                    tmp += texVal;

                    curValueHash = hashGHB(&ghb[my_ghb_index]);
                    texVal = tex1D(tref,curValueHash);
                    //updateGHB(&ghb[my_ghb_index],texVal);
                    tmp += texVal;
                     */

                    curValueHash = hashGHB(&ghb[my_ghb_index]);
                    //float texVal = tex1D(tref,((float)my_ghb_index / ((float) (blockDim.x * blockDim.y) * 3.0)) / 170.0);
                    texVal = tex1D(tref,curValueHash / 175.0);
                    tmp += texVal;
                    updateGHB(&ghb[my_ghb_index],texVal);

                    //tmp += ( input[curElement-1+shift] + input[curElement+1+shift] + input[curElement + singleRowOffset+shift]);// + input[curElement - singleRowOffset+shift] );
                    tmp /= 4.0;
                    output[curElement] = tmp;
                } // endif curelement in array
            } // end if jdx and idx in prop. range
        } // end for-jdx
    } // end for-idx
}

__global__ void avgMark(float* input, float* output, float* hashes, float* threadReads, uint width, uint height,int shift )
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

    // declare shared memory for per-thread value buffers
    extern __shared__ float ghb[];


    // still check for this in case of small img, not all threads need execute
    for (int idx = i; idx < (i + xScale); idx++) {
        if ( ((idx == i+xScale) && xmod == 0) ||
                ((idx == i+xScale) && width <= totalX )) break; // exact mult. corner case
        for (int jdx = j; jdx < (j + yScale); jdx++) { // over each element to proc
            if( ((jdx == j + yScale) && ymod == 0) ||
                    ((jdx == j + yScale) && height <= totalY)) break; // same corner case
            if( jdx < height-2 && jdx > 1
                    && idx < width-2 && idx > 1 ){
                float tmp = 0.0;
                int curElement = SINGLEDIMINDEX(jdx,idx,width);
               // int scaled = curElement * 4;
                int scaledGHB = ((threadIdx.y * blockDim.x) + threadIdx.x) * 3;
                int singleRowOffset = SINGLEDIMINDEX(1,0,width);
                if(curElement < (width*height) && curElement >= 0) {
                    /*
                    float loaded = input[curElement-1];
                    tmp += loaded;
                    float curHash = hashGHB(&(ghb[scaledGHB]));
                    //threadReads[scaled] = loaded;
                    //hashes[scaled] = curHash;
                    updateGHB(&(ghb[scaledGHB]),loaded);

                    loaded = input[curElement+1];
                    tmp += loaded;
                    curHash = hashGHB(&(ghb[scaledGHB]));
                    //threadReads[scaled+1] = loaded;
                    //hashes[scaled+1] = curHash;
                    updateGHB(&(ghb[scaledGHB]),loaded);

                    loaded = input[curElement-singleRowOffset];
                    tmp += loaded;
                    curHash = hashGHB(&(ghb[scaledGHB]));
                    //threadReads[scaled+2] = loaded;
                    //hashes[scaled+2] = curHash;
                    updateGHB(&(ghb[scaledGHB]),loaded);

                    loaded = input[curElement+singleRowOffset];
                    tmp += loaded;
                    curHash = hashGHB(&(ghb[scaledGHB]));
                    //threadReads[scaled+3] = loaded;
                    //hashes[scaled+3] = curHash;
                    updateGHB(&(ghb[scaledGHB]),loaded);
                       loaded = input[curElement+1];
                       tmp += loaded;
                       threadReads[scaled+1] = loaded;
                       loaded = input[curElement-singleRowOffset];
                       tmp += loaded;
                       threadReads[scaled+2] = loaded;
                       loaded = input[curElement+singleRowOffset];
                       tmp += loaded;
                       threadReads[scaled+3] = loaded;
                     */
                    tmp += ( input[curElement-1+shift] + input[curElement+1+shift] + input[curElement + singleRowOffset+shift] + input[curElement - singleRowOffset+shift] );
                    tmp /= 4.0;
                    output[curElement] = tmp;
                }
            }
        }
    }
}

// test for texture memory
int main(int argc, char* argv[])
{
    std::string inputTexFile;
    if(argc != 2) {
        std::cout << "Just give me a text file already...." << std::endl;
        exit(EXIT_FAILURE);
    } else {
        // turn argv[1] into a string
        inputTexFile.assign(argv[1]);
    }

    unsigned int cols = 1280;
    unsigned int rows = 720;

    cudaDeviceReset();
    cudaDeviceProp dev_props;
    // assume device 0
    HANDLE_ERROR( cudaGetDeviceProperties(&dev_props,0) );

    // print some stuff
    printf("Current Device compute capability: %d.%d\n",dev_props.major,dev_props.minor);
    printf("1D texture memory limit (cudaArray): %d\n",dev_props.maxTexture1D);
    /*
       printf("1D texture memory limit (linear mem): %d\n",dev_props.maxTexture1DLinear);
       printf("2D texture memory space limits: %d x %d\n",dev_props.maxTexture2D[0],dev_props.maxTexture2D[1]);
       printf("2D texture memory (gathered) space limits: %d x %d\n",dev_props.maxTexture2DGather[0],dev_props.maxTexture2DGather[1]);
     */
    // setup texture reference parameters
    tref.addressMode[0] = cudaAddressModeMirror; 
    tref.filterMode = cudaFilterModeLinear;
    tref.normalized = true; // access with coordinates in range [0-1)
#ifndef TEXMEM
    float* big_arr = (float*)malloc( rows*cols*sizeof(float)); 
    for(int i = 0;i<rows*cols;i++)
        big_arr[i] = (float)rand() / (float)RAND_MAX;
#else
    // create our big tex mem file reader
    LVAData tex_array_maker(inputTexFile);
    //tex_array_maker.PrintTables();
    // create a big set of values that represents our texture memory
    FloatVector tex_arr = tex_array_maker.GenerateUniqueValueSet(dev_props.maxTexture1D);

    // create a big array for input tex values
    float* big_arr = (float*)malloc( tex_arr.size() * sizeof(float)); 
    FloatVector::iterator it = tex_arr.begin();
    int i = 0;
    for(; it != tex_arr.end(); it++) {
        big_arr[i] = *it;
        i++;
    }
    printf("Total size of texture memory input: %d\n", tex_arr.size());
#endif


    float* out_arr = (float*)calloc(rows*cols,sizeof(float));
    float* reads = (float*)calloc(rows*cols*4,sizeof(float)); // this is for all values read by each thread (idx,jdx)
    float* hashes = (float*)calloc(rows*cols*4,sizeof(float)); // this is for all values read by each thread (idx,jdx)
    // alloc output
    float* gmemOutput;
    HANDLE_ERROR( cudaMalloc((void**)&gmemOutput,rows*cols*sizeof(float)) );


    // set up the thread block sizes
    int threadsX = 16, threadsY = 8;
    int rowsIn = floor((rows+1)/8);
    int colsIn = floor((cols+1)/8);
    int nBlocksWide = colsIn/threadsX;
    if (colsIn % threadsX) nBlocksWide++;
    int nBlocksTall = rowsIn/threadsY;
    if (rowsIn % threadsY) nBlocksTall++;
    dim3 nblocks(nBlocksWide,nBlocksTall);
    dim3 threadsPerBlock(threadsX,threadsY);

    cudaEvent_t kstart,kstop;
    HANDLE_ERROR( cudaEventCreate(&kstart) );
    HANDLE_ERROR( cudaEventCreate(&kstop) );
    float avg = 0.0;
    printf("Thread block dimensions: %dx%d. Num thread blocks: %d x %d\n",threadsX,threadsY,nBlocksWide,nBlocksTall);
#ifdef TEXMEM
    //printf("Executing with texture memory reads...\n");
    // set up global memory and texture reference

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
    cudaArray* my_arr;
    HANDLE_ERROR( cudaMallocArray(&my_arr,&channelDesc,dev_props.maxTexture1D,0,0) );

    // copy in
    HANDLE_ERROR( cudaMemcpyToArray(my_arr,0,0,big_arr,dev_props.maxTexture1D*sizeof(float),cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaBindTextureToArray(tref,my_arr,channelDesc) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    /*
       cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
       float* gmemInput;
       HANDLE_ERROR( cudaMalloc((void**)&gmemInput,tex_arr.size()*sizeof(float)) );
       HANDLE_ERROR( cudaMemcpy(gmemInput,&big_arr[0],tex_arr.size()*sizeof(float),cudaMemcpyHostToDevice) );

    // bind this texture reference to linear memory
    HANDLE_ERROR( cudaBindTexture((size_t) 0,tref,gmemInput,tex_arr.size()*sizeof(float)) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
     */
    srand(1);
    float* input_arr = (float*)malloc( rows*cols*sizeof(float)); 
    for(int i = 0;i<rows*cols;i++)
        input_arr[i] = (float)rand() / (float)RAND_MAX;

    float* gmemInput;
    HANDLE_ERROR( cudaMalloc((void**)&gmemInput,rows*cols*sizeof(float)) );
    HANDLE_ERROR( cudaMemcpy(gmemInput,&input_arr[0],tex_arr.size()*sizeof(float),cudaMemcpyHostToDevice) );

    int bytesForSmem = threadsX * threadsY * 3 * sizeof(float); // each thread gets 3 entries of 4 bytes each
    // compute
    for(int i =0;i<16;i++) {
        cudaEventRecord(kstart);
        texMark<<<nblocks,threadsPerBlock,bytesForSmem>>>(gmemInput,gmemOutput,cols,rows,0);
        HANDLE_ERROR( cudaGetLastError() );
        cudaEventRecord(kstop);
        HANDLE_ERROR( cudaEventSynchronize(kstop) );
        float time;
        cudaEventElapsedTime(&time,kstart,kstop);
        avg += time;
        printf("Kern exec. time (tex mem): %0.6f ms for shift = %d\n",time,i);
    }
    avg /= 16.0;

    // copy out
    HANDLE_ERROR( cudaMemcpy(out_arr,gmemOutput,rows*cols*sizeof(float),cudaMemcpyDeviceToHost) );
#else
    printf("Using boring global memory...\n");
    float* gmemInput;
    HANDLE_ERROR( cudaMalloc((void**)&gmemInput,rows*cols*sizeof(float)) );
    float* threadReads, *threadHashes;
    HANDLE_ERROR( cudaMalloc((void**)&threadReads,4*rows*cols*sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&threadHashes,4*rows*cols*sizeof(float)) );

    // copy in
    HANDLE_ERROR( cudaMemcpy(gmemInput,big_arr,rows*cols*sizeof(float),cudaMemcpyHostToDevice) );

    // compute
    int bytesForSmem = threadsX * threadsY * 3 * sizeof(float); // each thread gets 3 entries of 4 bytes each
    printf("Launching with %d bytes of smem, (or %d per thread)...\n",bytesForSmem,3*sizeof(float));
    for(int i=0;i<16;i++) {
        cudaEventRecord(kstart);
        avgMark<<<nblocks,threadsPerBlock,bytesForSmem>>>(gmemInput,gmemOutput,threadHashes,threadReads,cols,rows,0);
        HANDLE_ERROR( cudaGetLastError() );
        cudaEventRecord(kstop);
        HANDLE_ERROR(cudaEventSynchronize(kstop));
        float time;
        cudaEventElapsedTime(&time,kstart,kstop);
        avg += time;
        printf("Kern exec. time (glob mem): %0.6f ms for shift = %d\n",time,0);
    }
    avg /= 16.0;

    // copy out
    HANDLE_ERROR( cudaMemcpy(out_arr,gmemOutput,rows*cols*sizeof(float),cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(reads,threadReads,4*rows*cols*sizeof(float),cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(hashes,threadHashes,4*rows*cols*sizeof(float),cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaDeviceSynchronize() );

    //int startVal = 2*cols + 2;
    //int endVal = (rows-1)*cols - 2;
    /*
       for(int i = startVal;i < endVal;i+=4) {
       printf("Global history hash [%0.5f], next value: %0.5f\n",hashes[i],reads[i]);
       printf("Global history hash [%0.5f], next value: %0.5f\n",hashes[i+1],reads[i+1]);
       printf("Global history hash [%0.5f], next value: %0.5f\n",hashes[i+2],reads[i+2]);
       printf("Global history hash [%0.5f], next value: %0.5f\n",hashes[i+3],reads[i+3]);
       }
     */
#endif

    printf("Avg execution time: %0.6f ms\n",avg);

#ifdef TEXMEM
    cudaFreeArray(my_arr);
    free(input_arr);
#else
    cudaFree(threadReads);
    cudaFree(gmemInput);
#endif
    cudaFree(gmemOutput);
    free(big_arr);        
    free(out_arr);
    free(reads);
    free(hashes);
    return 0;
}
