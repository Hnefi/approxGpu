// Mark Sutherland, Josh San Miguel
//  - U of Toronto, 2015
//  - heavily derived from sd-vbs read image, just over-writes the pixels
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <ctype.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

#include "sdvbs_common.h"


#define IMG_DATA_OFFSET_POS 10
#define BITS_PER_PIXEL_POS 28

void writeImgToFile(F2D* imgB,F2D* imgG, F2D* imgR, const char* inputName, const char* outputName,bool singlechannel)
{
    // Reading BMP image
	int i;
	int fd;
	char *fdata;
	struct stat finfo;
	fd = open(inputName, O_RDONLY);
	fstat(fd, &finfo);

	fdata = (char*) malloc(finfo.st_size);
	
	read (fd, fdata, finfo.st_size);

	if ((fdata[0] != 'B') || (fdata[1] != 'M')) 
	{
		printf("File is not a valid bitmap file. Terminating the program\n");
        free(fdata);
		exit(1);
	}
	unsigned short *bitsperpixel = (unsigned short *)(&(fdata[BITS_PER_PIXEL_POS]));
    if(*bitsperpixel != 24) {
		printf("Error: Invalid bitmap format - ");
		printf("This application only accepts 24-bit pictures. Exiting\n");
        free(fdata);
		exit(1);
    }

	unsigned short *data_pos = (unsigned short *)(&(fdata[IMG_DATA_OFFSET_POS]));

	int imgdata_bytes = (int)finfo.st_size - (int)(*(data_pos));
	//printf("This file has %d bytes of image data, %d pixels\n", imgdata_bytes, imgdata_bytes / 3);

	int width = *((int*)&fdata[18]);
	int height = *((int*)&fdata[22]);
	int fileSize = (int) finfo.st_size;	

	//p will point to the first pixel
	char* p = &(fdata[*data_pos]);

    // iterate through each float and write it as an unsigned char..... UGLY
    int numPix = imgG->height * imgG->width;
    char* buf = (char*) malloc(numPix*3);
    int wdx = 0;
    for(int nI = (imgG->height)-1;nI >= 0; nI--) {
        for(int nJ = 0; nJ < imgG->width; nJ++) {
            int rdx = (nI * imgG->width) + nJ;
            int cast;
            if(!singlechannel) {
                cast = (int) imgB->data[rdx];
            } else {
                cast = 0;
            }
            assert( (unsigned char)cast <= 256 && (unsigned char) cast >= 0);
            buf[wdx++] = (unsigned char) cast;

            cast = (int) imgG->data[rdx];
            assert( (unsigned char)cast <= 256 && (unsigned char) cast >= 0);
            buf[wdx++] = (unsigned char)cast;

            if(!singlechannel) {
                cast = (int) imgR->data[rdx];
            } else {
                cast = 0;
            }
            assert( (unsigned char)cast <= 256 && (unsigned char) cast >= 0);
            buf[wdx++] = (unsigned char)cast;
        }
    }
    /*
    int j = 0;
    for(int i = 0; i < numPix*3; i+=3) {
        int cast = (int) imgB->data[j];
        assert( (char)cast < 255);
        buf[i] = (char) cast;

        cast = (int) imgG->data[j];
        assert( (char)cast < 255);
        buf[i+1] = (char)cast;

        cast = (int) imgR->data[j];
        assert( (char)cast < 255);
        buf[i+2] = (char)cast;

        j++;
    }
    */
    memcpy(&(fdata[*data_pos]),buf,numPix*3);

	FILE *writeFile; 
	writeFile = fopen(outputName,"w+");
	for(i = 0; i < fileSize; i++)
		fprintf(writeFile,"%c", fdata[i]);
	fclose(writeFile);
    
    free(buf);
    free(fdata);
    return;
}
