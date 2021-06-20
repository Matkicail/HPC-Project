#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../Helpers/fuzzyPoint.h"

#ifndef fuzzyMEANS
#define fuzzyMEANS

#define NUMPOINTS 100

// Need a 2nd set of eyes for this one.
// I believe it should work but it does not.
void calculateCentroids(FuzzyPoint *centroids, FuzzyPoint *data){
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        for(int j = 0 ; j < DIMENSIONS; j++){
            float probSum = 0.0f;
            float pointSum = 0.0f;
            for(int k = 0 ; k < NUMPOINTS ; k++){
                probSum += data[k].clusters[i];
                pointSum += data[k].values[j]  * data[k].clusters[i];
            }
            centroids[i].values[j] = pointSum / probSum;
        }
    }
}

void initAllFuzzyPoints(FuzzyPoint *data){
    for(int i = 0 ; i < NUMPOINTS ; i++){
        // can make a function to automate this
        initFuzzyPoint(&data[i]);
    }
}
void printCentroids(FuzzyPoint *centroids){
    printf("######################################################################################################\n");
    printf("Centroids \n");
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        printFuzzyPoint(centroids[i]);
    }
    printf("######################################################################################################\n");
}

void printAllFuzzyPoints(FuzzyPoint *data){
    printf("######################################################################################################\n");
    printf("Fuzzy Data Points \n");
    for(int i = 0 ; i < NUMPOINTS ; i++){
        printFuzzyPoint(data[i]);
    }
    printf("######################################################################################################\n");
}

#endif