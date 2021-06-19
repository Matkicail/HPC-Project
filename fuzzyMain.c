#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "fuzzyMeans.h"

void testFuzzy();

int main(){
    testFuzzy();
    return 0;
}

void testFuzzy(){
    FuzzyPoint x,y,z;
    initPoint(&x);
    initPoint(&y);
    initPoint(&z);
    printPoint(x);
    printPoint(y);
    printPoint(z);
}