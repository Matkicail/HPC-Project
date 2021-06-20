#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "fuzzyMeans.h"

void testFuzzy();

int main(){

    // SERIAL CODE SEGMENT
    // testFuzzy();
    FuzzyPoint *centroids = (FuzzyPoint *)malloc(NUMCLUSTER * sizeof(FuzzyPoint));
    FuzzyPoint *data = (FuzzyPoint *)malloc(NUMPOINTS * sizeof(FuzzyPoint));
    initAllFuzzyPoints(data);
    printAllFuzzyPoints(data);
    initCentroids(centroids);
    // calculateCentroids(centroids, data);
    printCentroids(centroids);

    // SERIAL CODE SEGMENT

    // PARALLEL CODE SEGMENT

    // PARALLEL CODE SEGMENT

    // Freeing of memory
    free(centroids);
    free(data);
    printf("success \n");
    return 0;
}

void testFuzzy(){
    
}