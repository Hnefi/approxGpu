/********************************
Author: Sravanthi Kota Venkata
********************************/

#include <stdio.h>
#include <stdlib.h>
#include "sdvbs_common.h"

void fWriteMatrix(F2D* input, char* inpath)
{
    FILE* fp;
    char im[100];
    int rows,cols, i, j;

#ifndef APPROXIMATE
    sprintf(im, "%s/expected_C.txt", inpath);
#endif

#ifdef APPROXIMATE
    sprintf(im, "%s/approx_C.txt",inpath);
#endif
    fp = fopen(im, "w");

    //printf("Got into writing output matrix to file %s\n.",im);
    rows = input->height;
    cols = input->width;

    for(i=0; i<rows; i++)
    {
        for(j=0; j<cols; j++)
        {
            fprintf(fp, "%f\t", subsref(input, i, j));
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}



