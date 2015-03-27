#include "sdvbs_common.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <math.h>

using std::size_t;
using std::cout;
using std::endl;

void pixDiff(const char* gold, const char* approx)
{
    I2D* goldChannels[3];
    I2D* approxChannels[3];
    I2D* Ig,*Ia;
    int rows, cols;
    for(int arg=2;arg>=0;arg--) {
        Ig = readImage(gold,arg);
        Ia = readImage(approx,arg);
        rows = Ig->height;
        cols = Ig->width;
        // copy each image channel into the "saved pixels" to compute later
        goldChannels[arg] = iDeepCopy(Ig);
        approxChannels[arg] = iDeepCopy(Ia);
        iFreeHandle(Ig);
        iFreeHandle(Ia); 
    }

    // now for each element, compute difference (relative to 255 max value of each channel)
    float total = 0.0;
    for(int i=0;i<rows*cols;i++) {
        int rdiff = abs( goldChannels[0]->data[i] - approxChannels[0]->data[i] );
        int gdiff = abs( goldChannels[1]->data[i] - approxChannels[1]->data[i] );
        int bdiff = abs( goldChannels[2]->data[i] - approxChannels[2]->data[i] );
        float thispdiff = (float)rdiff + (float)gdiff + (float)bdiff;
        thispdiff /= 3.0;
        total += thispdiff;
    }
    float numelem = (float)rows*(float)cols;
    total /= numelem;

    cout << "Mean pixel diff: " << total << endl;
}
