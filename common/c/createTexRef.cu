#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "../kernels/texRef.h"
#include "sdvbs_common.h"

// sets up the global texture reference with allocation and release

bool createTextureReference(int rows, int cols, std::string inFile) {

    cudaDeviceProp dev_props;
    // assume device 0
    HANDLE_ERROR( cudaGetDeviceProperties(&dev_props,0) );

    // print some stuff
    printf("Current Device compute capability: %d.%d\n",dev_props.major,dev_props.minor);
    printf("1D texture memory limit (cudaArray): %d\n",dev_props.maxTexture1D);

    // setup texture reference parameters
    tref.addressMode[0] = cudaAddressModeMirror; 
    tref.filterMode = cudaFilterModeLinear;
    tref.normalized = true; // access with coordinates in range [0-1)

    // read in the texture training set from the input file
    float* big_arr = (float*)malloc( dev_props.maxTexture1D * sizeof(float)); 
    std::ifstream inputStream(inFile.c_str());
    if ( !inputStream.is_open() ) return false;
    std::string raw_string;
    int i = 0;
    while( !inputStream.eof() ) {
        getline(inputStream,raw_string);
        float value = atof(raw_string.c_str());
        big_arr[i] = value;
        i++;
    }
    printf("Total size of training input read from file: %d\n", i);

    // setup cudaArray and place this into the device.
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
    cudaArray* my_arr; // my_arr is defined in ../kernels/texRef.h
    HANDLE_ERROR( cudaMallocArray(&my_arr,&channelDesc,i,0,0) );

    // copy in
    HANDLE_ERROR( cudaMemcpyToArray(my_arr,0,0,big_arr,i*sizeof(float),cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaBindTextureToArray(tref,my_arr,channelDesc) );
    HANDLE_ERROR( cudaDeviceSynchronize() );

    return true;
}
