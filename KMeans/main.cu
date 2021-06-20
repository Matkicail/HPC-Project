#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "./Helpers/point.h"
#include "./Headers/kMeans.h"
#include "./Headers/kMeans.cuh"


#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// // Utilities and timing functions
// #include "helper_functions.h"    // includes cuda.h and cuda_runtime_api.h

// // CUDA helper functions
// #include <helper_cuda.h>         // helper functions for CUDA error check

#define numBlocks 
#define numThreads

void testPoints();

int main(){
    
    // Serial Code
    Point *kPoints = (Point *)malloc(NUMCLUSTER * sizeof(Point));
    initKPoints(kPoints);
    printCentroids(kPoints);
    Point *data = (Point *)malloc(NUMPOINTS * sizeof(Point));
    initDataPoints(data);
    // for(int i = 0 ; i < ITTERATIONS ; i++){
    //     assignDataCluster(kPoints, data);
    // }
    // printCentroids(kPoints);
    // Serial Code
    
    // Parallel Code

    // dim3 block();
    // dim3 threads();

    // Parallel Code


    free(kPoints);
    free(data);
    return 0;
}

// just a simple test I made to see if the points init
void testPoints(){
    Point x,y;
    initPoint(&y);
    initPoint(&x);
    printPoint(y);
    printPoint(x);
    printf("Distance %f \n",pointDistance(x,y));
    return;
}

