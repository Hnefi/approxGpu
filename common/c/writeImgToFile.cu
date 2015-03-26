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

void writeImgToFile(F2D* imgOut, const char* inputName, const char* outputName)
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
	printf("This file has %d bytes of image data, %d pixels\n", imgdata_bytes, imgdata_bytes / 3);

	int width = *((int*)&fdata[18]);
	int height = *((int*)&fdata[22]);
	int fileSize = (int) finfo.st_size;	

	//p will point to the first pixel
	char* p = &(fdata[*data_pos]);

    // iterate through each float and write it as an unsigned char..... UGLY
    int numPix = imgOut->height * imgOut->width;
    char* buf = (char*) malloc(numPix*3);
    int j = 0;
    for(int i = 0; i < numPix*3; i+=3) {
        buf[i] = (char) 0;
        int cast = (int) imgOut->data[j++];
        assert( (char)cast < 255);
        buf[i+1] = (char)cast;
        buf[i+2] = (char)0;
    }
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
