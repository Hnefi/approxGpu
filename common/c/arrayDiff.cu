#include "sdvbs_common.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <math.h>

using std::size_t;
using std::cout;
using std::endl;

float arrayDiff(I2D* gold, I2D* approx)
{
    float total = 0.0;
    int rows = gold->height;
    int cols = gold->width;
    for(int i=0;i<rows*cols;i++) {
        float gdiff = abs( (float)gold->data[i] - (float)approx->data[i] );
        total += gdiff;
    }
    float numelem = (float)rows*(float)cols;
    total /= numelem;
    return total;
}
