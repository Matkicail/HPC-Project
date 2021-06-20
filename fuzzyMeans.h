#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "fuzzyPoint.h"

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
                probSum += data[k].clusters[j];
                pointSum += data[k].values[j];
            }
            centroids[i].values[j] = pointSum / probSum;
        }
    }
}

void printCentroids(FuzzyPoint *centroids){
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        printFuzzyPoint(centroids[i]);
    }
}

#endif