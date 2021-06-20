#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "./Headers/fuzzyMeans.h"

void testFuzzy();

int main(){

    // SERIAL CODE SEGMENT
    // testFuzzy();
    FuzzyPoint *centroids = (FuzzyPoint *)malloc(NUMCLUSTER * sizeof(FuzzyPoint));
    FuzzyPoint *data = (FuzzyPoint *)malloc(NUMPOINTS * sizeof(FuzzyPoint));
    initAllFuzzyPoints(data);
    printAllFuzzyPoints(data);
    initCentroids(centroids);
    for(int i = 0 ; i < 1000 ; i++){
        calculateCentroids(centroids, data);
        // printCentroids(centroids);
        updateDataAssignment(centroids,data);
    }
    printf("CALCULATION FINISHED \n");
    printCentroids(centroids);
    printAllFuzzyPoints(data);
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