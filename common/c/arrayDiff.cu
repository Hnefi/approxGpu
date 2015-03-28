#include "sdvbs_common.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <math.h>

using std::size_t;
using std::cout;
using std::endl;

float arrayDiff(F2D* gold, F2D* approx)
{
    float total = 0.0;
    int rows = gold->height;
    int cols = gold->width;
    for(int i=0;i<rows*cols;i++) {
        int gdiff = abs( gold->data[i] - approx->data[i] );
        total += gdiff;
    }
    float numelem = (float)rows*(float)cols;
    total /= numelem;
    return total;
}
