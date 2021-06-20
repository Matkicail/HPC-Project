#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "fuzzyMeans.h"

void testFuzzy();

int main(){
    // testFuzzy();
    FuzzyPoint *centroids = (FuzzyPoint *)malloc(NUMCLUSTER * sizeof(FuzzyPoint));
    FuzzyPoint *data = (FuzzyPoint *)malloc(NUMPOINTS * sizeof(FuzzyPoint));
    initAllFuzzyPoints(data);
    printAllFuzzyPoints(data);
    initCentroids(centroids);
    // calculateCentroids(centroids, data);
    printCentroids(centroids);
    free(centroids);
    free(data);
    printf("success \n");
    return 0;
}

void testFuzzy(){
    
}