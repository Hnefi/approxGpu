/********************************
Author: Sravanthi Kota Venkata
 ********************************/

#include "tracking.h"
#include <iostream>
#include <fstream>
#include <string>

// LVA for interacting with pin
#ifdef APPROXIMATE
extern void LVA_FUNCTION(int type,void* start, void* end, int self) __attribute__ ((noinline));
extern void LVA_FUNCTION_RM(int type, void* start,void*end, int self) __attribute__ ((noinline));

extern void LVA_FUNCTION(int type, void* start, void* end, int self)
{ __asm__ __volatile__ ("xchg %dx,%dx"); }

extern void LVA_FUNCTION_RM(int type,void* start,void* end, int self)
{ __asm__ __volatile__ ("xchg %dx,%dx"); }
#endif

#define LVA_BX_INSTRUCTION __asm__ __volatile__ ("xchg %bx,%bx");

int main(int argc, char* argv[])
{
    int i, j, k, N_FEA, WINSZ, LK_ITER, rows, cols;
    int endR, endC;
    F2D *blurredImage, *previousFrameBlurred_level1, *previousFrameBlurred_level2, *blurred_level1, *blurred_level2;
    F2D *verticalEdgeImage, *horizontalEdgeImage, *verticalEdge_level1, *verticalEdge_level2, *horizontalEdge_level1, *horizontalEdge_level2, *interestPnt;
    F2D *lambda, *lambdaTemp, *features;
    I2D *Ic, *status;
    float SUPPRESION_RADIUS;
    F2D *newpoints;

    int numFind, m, n;
    F2D *np_temp;

    unsigned int* start, *end, *elapsed, *elt;
    char im1[100];
    int counter=2;
    float accuracy = 0.03;
    int count;

    if(argc < 3) 
    {
        printf("We need input image path AND training set file.\n");
        return -1;
    }
    std::string inputTexFile;
    sprintf(im1, "%s/bug_frames/1.bmp", argv[1]);
    char img1Name[100];
    sprintf(img1Name,"%s/bug_frames/1.bmp",argv[1]);
    inputTexFile.assign(argv[2]);

    N_FEA = 1600;
    WINSZ = 4;
    SUPPRESION_RADIUS = 10.0;
    LK_ITER = 20;

#ifdef test
    WINSZ = 2;
    N_FEA = 100;
    LK_ITER = 2;
    counter = 2;
    accuracy = 0.1;
#endif
#ifdef sim_fast
    WINSZ = 2;
    N_FEA = 100;
    LK_ITER = 2;
    counter = 4;
#endif
#ifdef sim
    WINSZ = 2;
    N_FEA = 200;
    LK_ITER = 2;
    counter = 4;
#endif
#ifdef sqcif
    WINSZ = 8;
    N_FEA = 500;
    LK_ITER = 15;
    counter = 2;
#endif
#ifdef qcif
    WINSZ = 12;
    N_FEA = 400;
    LK_ITER = 15;
    counter = 4;
#endif
#ifdef cif
    WINSZ = 20;
    N_FEA = 500;
    LK_ITER = 20;
    counter = 4;
#endif
#ifdef vga
    WINSZ = 32;
    N_FEA = 400;
    LK_ITER = 20;
    counter = 4;
#endif
#ifdef wuxga
    WINSZ = 64;
    N_FEA = 500;
    LK_ITER = 20;
    counter = 4;
#endif
#ifdef fullhd
    WINSZ = 48;
    N_FEA = 500;
    LK_ITER = 20;
    counter = 4;
#endif

    cudaDeviceReset();
    printf("Input size\t\t- (%dx%d)\n", rows, cols);

    cudaDeviceProp dev_props;
    // assume device 0
    HANDLE_ERROR( cudaGetDeviceProperties(&dev_props,0) );

    // print some stuff
    printf("Current Device compute capability: %d.%d\n",dev_props.major,dev_props.minor);
    printf("1D texture memory limit (cudaArray): %d\n",dev_props.maxTexture1D);

    // read in the texture training set from the input file
    float* big_arr = (float*)malloc( dev_props.maxTexture1D * sizeof(float)); 
    std::ifstream inputStream(inputTexFile.c_str());
    if ( !inputStream.is_open() ) return false;
    std::string raw_string;
    i = 0;
    while( !inputStream.eof() ) {
        getline(inputStream,raw_string);
        float value = atof(raw_string.c_str());
        //cout << "value: " << value << endl;
        big_arr[i] = value;
        i++;
    }
    printf("Total size of training input read from file: %d\n", i-1);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); 
    cudaArray* cuArray; 
    cudaMallocArray(&cuArray, &channelDesc, i-1); 
    // Copy to device memory some data located at address h_data in host memory 
    cudaMemcpyToArray(cuArray, 0, 0, big_arr,(i-1)*sizeof(float),cudaMemcpyHostToDevice); 
    // Specify texture struct 
    cudaResourceDesc resDesc; 
    memset(&resDesc, 0, sizeof(resDesc)); 
    resDesc.resType = cudaResourceTypeArray; 
    resDesc.res.array.array = cuArray; 
    // Specify texture object parameters 
    struct cudaTextureDesc texDesc; 
    memset(&texDesc, 0, sizeof(texDesc)); 
    texDesc.addressMode[0] = cudaAddressModeMirror; 
    texDesc.filterMode = cudaFilterModeLinear; 
    texDesc.readMode = cudaReadModeElementType; 
    texDesc.normalizedCoords = 1; 
    // Create texture object 
    cudaTextureObject_t texObj = 0; 
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);


    F2D* blurs[3];
    F2D* resizes[3];
    F2D* sobelx[3];
    F2D* sobely[3];

    // do this for all the rgb channels
    for(int arg = 2;arg >= 0;arg--) {
        Ic = readImage(im1,arg);
        rows = Ic->height;
        cols = Ic->width;
        ImagePyramid* preprocessed = createOutputImages(Ic,&texObj); // just need to define a struct to return 4 float* arrays
        //printf("After calling createImgPyramid...\n");
        //writeImgToFile(blurredImage,img1Name,"test.bmp");

        /** Scale down the image to build Image Pyramid. We find features across all scales of the image **/
        blurred_level1 = preprocessed->blurredImg;                   /** Scale 0 **/
        blurred_level2 = preprocessed->resizedImg;     /** Scale 1 **/
        horizontalEdgeImage = preprocessed->horizEdge;
        verticalEdgeImage = preprocessed->vertEdge;

        // copy the first image into the "saved pixels" to merge later
        blurs[arg] = fDeepCopy(blurred_level1);
        resizes[arg] = fDeepCopy(blurred_level2);
        sobelx[arg] = fDeepCopy(horizontalEdgeImage);
        sobely[arg] = fDeepCopy(verticalEdgeImage);
        destroyImgPyramid(preprocessed,0x0);
    }

    // write out the first image frame.
    writeImgToFile(blurs[2],blurs[1],blurs[0],img1Name,"blur_out.bmp");
    writeImgToFile(resizes[2],resizes[1],resizes[0],img1Name,"resize_out.bmp");
    writeImgToFile(sobelx[2],sobelx[1],sobelx[0],img1Name,"sobelx_out.bmp");
    writeImgToFile(sobely[2],sobely[1],sobely[0],img1Name,"sobely_out.bmp");

    /** Read input image **/
    Ic = readImage(im1);
    rows = Ic->height;
    cols = Ic->width;
    /* Other frames */
#define MAX_COUNTER     (4)
    I2D *Ics[MAX_COUNTER];
    ImagePyramid* newFramePyramids[MAX_COUNTER];
    cudaStream_t frameStreams[MAX_COUNTER];

    /** Until now, we processed base frame. The following for loop processes other frames **/
    for(count=1; count<=counter; count++)
    {
        /** Read image **/
        sprintf(im1, "%s/bug_frames/%d.bmp", argv[1], count);
        Ics[count-1] = readImage(im1);
    }

    //start roi
    LVA_BX_INSTRUCTION;
    LVA_BX_INSTRUCTION;

    /** Start Timing **/
    start = photonStartTiming();

    /** IMAGE PRE-PROCESSING **/

    /** Blur the image to remove noise - weighted avergae filter **/

    ImagePyramid* preprocessed = createImgPyramid(Ic, 0,&texObj,false); // just need to define a struct to return 4 float* arrays
    //printf("After calling createImgPyramid...\n");

    blurredImage = preprocessed->blurredImg;
    //writeImgToFile(blurredImage,img1Name,"test.bmp");

    /** Scale down the image to build Image Pyramid. We find features across all scales of the image **/
    blurred_level1 = fDeepCopy(preprocessed->blurredImg);                   /** Scale 0 **/
    blurred_level2 = fDeepCopy(preprocessed->resizedImg);     /** Scale 1 **/
    horizontalEdgeImage = preprocessed->horizEdge;
    verticalEdgeImage = preprocessed->vertEdge;

    // copy the first image into the "saved pixels" to merge later

    /** Edge images are used for feature detection. So, using the verticalEdgeImage and horizontalEdgeImage images, we compute feature strength
      across all pixels. Lambda matrix is the feature strength matrix returned by calcGoodFeature **/

    lambda = calcGoodFeature(verticalEdgeImage, horizontalEdgeImage, verticalEdgeImage->width, verticalEdgeImage->height, WINSZ);
    endR = lambda->height;
    endC = lambda->width;
    lambdaTemp = fReshape(lambda, endR*endC, 1);

    /** We sort the lambda matrix based on the strengths **/
    /** Fill features matrix with top N_FEA features **/
    fFreeHandle(lambdaTemp);
    lambdaTemp = fillFeatures(lambda, N_FEA, WINSZ);
    features = fTranspose(lambdaTemp);

    /** Suppress features that have approximately similar strength and belong to close neighborhood **/
    interestPnt = getANMS(features, SUPPRESION_RADIUS);

    /** Refill interestPnt in features matrix **/
    fFreeHandle(features);
    features = fSetArray(2, interestPnt->height, 0);
    for(i=0; i<2; i++) {
        for(j=0; j<interestPnt->height; j++) {
            subsref(features,i,j) = subsref(interestPnt,j,i); 
        }
    } 
    /* commented out these frees to perform one big batch free on the returned image structure
       fFreeHandle(verticalEdgeImage);
       fFreeHandle(horizontalEdgeImage);
     */
    fFreeHandle(interestPnt);
    fFreeHandle(lambda);
    fFreeHandle(lambdaTemp);
    iFreeHandle(Ic);
    destroyImgPyramid(preprocessed,0x0);

    /** Until now, we processed base frame. The following for loop processes other frames **/
    for(count=1; count<=counter; count++)
    {
        newFramePyramids[count-1] = createImgPyramid(Ics[count-1],0,&texObj,false);

        Ic = Ics[count-1];
        rows = Ic->height;
        cols = Ic->width;

        //printf("Read image %d of dim %dx%d.\n",count,rows,cols);
        /* Start timing */
        //start = photonStartTiming();


        /** Blur image to remove noise **/
        blurredImage = newFramePyramids[count-1]->blurredImg;

        /** Blur image to remove noise **/
        previousFrameBlurred_level1 = fDeepCopy(blurred_level1);
        previousFrameBlurred_level2 = fDeepCopy(blurred_level2);

        //MARK - added these because i deep copied into previousFrame, and then can get rid of the old
        fFreeHandle(blurred_level1);
        fFreeHandle(blurred_level2);

        /** Image pyramid **/
        blurred_level1 = fDeepCopy(blurredImage);
        blurred_level2 = fDeepCopy(newFramePyramids[count-1]->resizedImg);

        verticalEdge_level1 = newFramePyramids[count-1]->vertEdge;
        verticalEdge_level2 = newFramePyramids[count-1]->vertEdge_small;
        horizontalEdge_level1 = newFramePyramids[count-1]->horizEdge;
        horizontalEdge_level2 = newFramePyramids[count-1]->horizEdge_small;

        newpoints = fSetArray(2, features->width, 0);

        /** Based on features computed in the previous frame, find correspondence in the current frame. "status" returns the index of corresponding features **/
        status = calcPyrLKTrack(previousFrameBlurred_level1, previousFrameBlurred_level2, verticalEdge_level1, verticalEdge_level2, horizontalEdge_level1, horizontalEdge_level2, blurred_level1, blurred_level2, features, features->width, WINSZ, accuracy, LK_ITER, newpoints);

        destroyImgPyramid(newFramePyramids[count-1], count);

        // left these ones (because they were just alloc'd in this loop
        fFreeHandle(previousFrameBlurred_level1);
        fFreeHandle(previousFrameBlurred_level2);

        /** Populate newpoints with features that had correspondence with previous frame features **/
        np_temp = fDeepCopy(newpoints);
        if(status->width > 0 )
        {
            k = 0;
            numFind=0;
            for(i=0; i<status->width; i++)
            {
                if( asubsref(status,i) == 1)
                    numFind++;
            }
            fFreeHandle(newpoints);
            newpoints = fSetArray(2, numFind, 0);

            for(i=0; i<status->width; i++)
            {
                if( asubsref(status,i) == 1)
                {
                    subsref(newpoints,0,k) = subsref(np_temp,0,i);
                    subsref(newpoints,1,k++) = subsref(np_temp,1,i);
                }
            }    
        }    

        iFreeHandle(status);
        iFreeHandle(Ic);
        fFreeHandle(np_temp);
        fFreeHandle(features);
        /** Populate newpoints into features **/
        features = fDeepCopy(newpoints);
        //printf("Printing features...\n");
        /*for(i = 0;i<features->height;i++) {
          for(j=0;j<features->width;j++) {
          printf("%f\t",subsref(features,i,j));
          }
          printf("\n");
          }*/

        fFreeHandle(newpoints);
    }
    /* Timing utils */
    end = photonEndTiming();
    elapsed = photonReportTiming(start, end);
    free(start);
    free(end);   

    //end roi
    LVA_BX_INSTRUCTION;


#ifdef CHECK   
    /* Self checking */
    {
        int ret=0;
        float tol = 2.0;
#ifdef GENERATE_OUTPUT
        fWriteMatrix(features, argv[1]);
#endif
        ret = fSelfCheck(features, argv[1], tol); 
        if (ret == -1)
            printf("Error in Tracking Map\n");
    }
#endif

    photonPrintTiming(elapsed);

    fFreeHandle(blurred_level1);
    fFreeHandle(blurred_level2);
    fFreeHandle(features);

    free(elapsed);
    free(big_arr);
    //free texture reference
    cudaFreeArray(cuArray);
    cudaDeviceReset();
    return 0;
}
