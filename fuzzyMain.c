#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "fuzzyMeans.h"

void testFuzzy();

int main(){
    // testFuzzy();
    FuzzyPoint *data = (FuzzyPoint *)malloc(NUMPOINTS * sizeof(FuzzyPoint));
    FuzzyPoint *centroids = (FuzzyPoint *)malloc(NUMCLUSTER * sizeof(FuzzyPoint));
    initCentroids(centroids);
    initFuzzyPoints(data, NUMPOINTS);
    printFuzzyPoints(centroids,NUMCLUSTER);
    printf("Calculating \n");
    calculateCentroids(centroids, data, NUMPOINTS);
    printFuzzyPoints(centroids,NUMCLUSTER);
    printf("dead ? \n");
    free(centroids);
    free(data);
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