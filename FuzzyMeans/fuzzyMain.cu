#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "./Headers/fuzzyMeans.h"

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
// Wait for Jared on this one, something really weird is going on in the image helper & we do not need that file at all.
// #include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include "./Headers/fuzzyMeans.cuh"

int main(){

    // SERIAL CODE SEGMENT
    // testFuzzy();
    FuzzyPoint *centroids = (FuzzyPoint *)malloc(NUMCLUSTER * sizeof(FuzzyPoint));
    FuzzyPoint *data = (FuzzyPoint *)malloc(NUMPOINTS * sizeof(FuzzyPoint));
    initAllFuzzyPoints(data);
    printAllFuzzyPoints(data);
    initCentroids(centroids);

    for(int i = 0 ; i < NUMCLUSTER ; i++){
        initValues(&centroids[i]);
    }
    printAllFuzzyPoints(data);

    // // SERIAL CODE SEGMENT
    // for(int i = 0 ; i < 100 ; i++){
    //     calculateCentroids(centroids, data);
    //     // printCentroids(centroids);
    //     updateDataAssignment(centroids,data);
    // }
    // printf("CALCULATION FINISHED \n");
    // printCentroids(centroids);
    

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