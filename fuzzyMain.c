#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "fuzzyMeans.h"

int main(){
    FuzzyPoint x,y,z;
    initPoint(&x);
    initPoint(&y);
    initPoint(&z);
    printPoint(x);
    printPoint(y);
    printPoint(z);
    return 0;
}