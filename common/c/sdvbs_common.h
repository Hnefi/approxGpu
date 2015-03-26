/********************************
Author: Sravanthi Kota Venkata
********************************/

#ifndef _SDVBS_COMMON_
#define _SDVBS_COMMON_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>

typedef struct
{
    int width;
    int height;
    int data[];
    // THIS SHIT BELOW IS ALL CUDA POINTERS>>>>
    // WHO WRITES CODE THIS WAY.>>>>>>>>>>> 
    // FAIL
    int* d_inputPixels;
    float* d_outputPixels;
    float* d_intermediate;
    int* d_weightedKernel,*sobel_kern_1,*sobel_kern_2;
    float* resizeInt, *dxInt, *dyInt, *dyInt_small, *dxInt_small;
    float* resizeOutput, *dxOutput, *dyOutput, *dxOutput_small, *dyOutput_small;
}I2D;

typedef struct
{
    int width;
    int height;
    unsigned int data[];
}UI2D;

typedef struct
{
    int width;
    int height;
    float data[];
}F2D;

typedef struct {
    F2D* blurredImg;
    F2D* resizedImg;
    F2D* horizEdge;
    F2D* vertEdge;
    F2D* horizEdge_small;
    F2D* vertEdge_small;
} ImagePyramid;

typedef struct {
    F2D* final;
    F2D* intermediate;
} TwoStepKernel;

#define subsref(a,i,j) a->data[(i) * a->width + (j)]
#define asubsref(a,i) a->data[i]
#define arrayref(a,i) a[i]

static void HandleError( cudaError_t err,
        const char *file,
        int line ) {
    if (err != cudaSuccess) {
        printf( "--RETURN ERROR--: %s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/** Image read and write **/
I2D* readImage(const char* pathName,int rgb = 1);
F2D* readFile(unsigned char* fileName);
void writeImgToFile(F2D* imgR,F2D* imgB, F2D* imgG,const char* inName, const char* outName);


/** Memory allocation functions **/
I2D* iMallocHandle(int rows, int cols);
F2D* fMallocHandle(int rows, int cols);
UI2D* uiMallocHandle(int rows, int cols);

void iFreeHandle(I2D* out);
void fFreeHandle(F2D* out);
void uiFreeHandle(UI2D* out);

/** Memory copy/set function **/
I2D* iSetArray(int rows, int cols, int val);
F2D* fSetArray(int rows, int cols, float val);
I2D* iDeepCopy(I2D* in);
F2D* fDeepCopy(F2D* in);
I2D* iDeepCopyRange(I2D* in, int startRow, int numberRows, int startCol, int numberCols);
F2D* fDeepCopyRange(F2D* in, int startRow, int numberRows, int startCol, int numberCols);
F2D* fiDeepCopy(I2D* in);
I2D* ifDeepCopy(F2D* in);


/** Matrix operations - concatenation, reshape **/
F2D* ffVertcat(F2D* matrix1, F2D* matrix2);
I2D* iVertcat(I2D* matrix1, I2D* matrix2);
F2D* fHorzcat(F2D* a, F2D* b);
I2D* iHorzcat(I2D* a, I2D* b);
F2D* horzcat(F2D* a, F2D* b, F2D* c);
F2D* fTranspose(F2D* a);
I2D* iTranspose(I2D* a);
F2D* fReshape(F2D* in, int rows, int cols);
I2D* iReshape(I2D* in, int rows, int cols);


/** Binary Operations **/
F2D* fDivide(F2D* a, float b);
F2D* fMdivide(F2D* a, F2D* b);
F2D* ffDivide(F2D* a, F2D* b);
F2D* ffTimes(F2D* a, float b);
F2D* fTimes(F2D* a, F2D* b);
I2D* iTimes(I2D* a, I2D* b);
F2D* fMtimes(F2D* a, F2D* b);
F2D* ifMtimes(I2D* a, F2D* b);
F2D* fMinus(F2D* a, F2D* b);
I2D* iMinus(I2D* a, I2D* b);
I2D* isMinus(I2D* a, int b);
F2D* fPlus(F2D* a, F2D* b);
I2D* isPlus(I2D* a, int b);


/** Filtering operations **/
F2D* calcSobel_dX(F2D* imageIn);
//F2D* calcSobel_dY(F2D* imageIn);
TwoStepKernel* calcSobel_dY(F2D* imageIn);
F2D* ffConv2(F2D* a, F2D* b);
F2D* fiConv2(I2D* a, F2D* b);
F2D* ffConv2_dY(F2D* a, F2D* b);
F2D* ffiConv2(F2D* a, I2D* b);
I2D* iiConv2(I2D* a, I2D* b);


/** Image Transformations - resize, integration etc **/
ImagePyramid* createImgPyramid(I2D* imageIn, cudaStream_t d_stream, cudaTextureObject_t* tref,bool train_set);
ImagePyramid* createOutputImages(I2D* imageIn, cudaTextureObject_t* tref);

bool createTextureReference(int rows, int cols, std::string inFile);
void destroyImgPyramid(ImagePyramid* retStruct,int imgNum);
TwoStepKernel* imageResize(F2D* imageIn);
TwoStepKernel* imageBlur(I2D* imageIn);


/** Support functions **/
F2D* fFind3(F2D* in);
F2D* fSum2(F2D* inMat, int dir);
F2D* fSum(F2D* inMat);
I2D* iSort(I2D* in, int dim);
F2D* fSort(F2D* in, int dim);
I2D* iSortIndices(I2D* in, int dim);
I2D* fSortIndices(F2D* input, int dim);
F2D* randnWrapper(int m, int n);
F2D* randWrapper(int m, int n);


/** Checking functions **/
int selfCheck(I2D* in1, char* path, int tol);
int fSelfCheck(F2D* in1, char* path, float tol);
void writeMatrix(I2D* input, char* inpath);
void fWriteMatrix(F2D* input, char* inpath);


/** Timing functions **/
unsigned int* photonEndTiming();
unsigned int* photonStartTiming();
unsigned int* photonReportTiming(unsigned int* startCycles,unsigned int* endCycles);
void photonPrintTiming(unsigned int * elapsed);


#endif

