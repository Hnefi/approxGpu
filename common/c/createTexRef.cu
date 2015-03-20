#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "sdvbs_common.h"

using std::cout;
using std::endl;

// sets up the global texture reference with allocation and release
bool createTextureReference(int rows, int cols, std::string inFile) {
#if 0

    // setup texture reference parameters
    tref.addressMode[0] = cudaAddressModeClamp; 
    tref.filterMode = cudaFilterModePoint;
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
        //cout << "value: " << value << endl;
        big_arr[i] = value;
        i++;
    }
    printf("Total size of training input read from file: %d\n", i);

    // setup cudaArray and place this into the device.
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
    //cudaArray* my_arr; // my_arr is defined in ../kernels/texRef.h
    HANDLE_ERROR( cudaMallocArray(&my_arr,&channelDesc,i) );

    // copy in
    HANDLE_ERROR( cudaMemcpyToArray(my_arr,0,0,big_arr,i*sizeof(float),cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaBindTextureToArray(tref,my_arr,channelDesc) );

    float* copy_out = (float*)malloc( 100*sizeof(float));
    HANDLE_ERROR( cudaMemcpyFromArray(copy_out,my_arr,0,0,3*sizeof(float),cudaMemcpyDeviceToHost) );
    printf("%f, %f, %f\n",copy_out[0],copy_out[1],copy_out[2]);
    free(copy_out);
#endif
    return true;
}
