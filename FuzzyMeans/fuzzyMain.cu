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
    // printAllFuzzyPoints(data);
    initCentroids(centroids);

    for(int i = 0 ; i < NUMCLUSTER ; i++){
        initValues(&centroids[i]);
    }
    // printAllFuzzyPoints(data);
    // // SERIAL CODE SEGMENT
    // for(int i = 0 ; i < 100 ; i++){
    //     calculateCentroids(centroids, data);
    //     // printCentroids(centroids);
    //     updateDataAssignment(centroids,data);
    // }
    // printf("CALCULATION FINISHED \n");
    // printCentroids(centroids);
    

    // PARALLEL CODE SEGMENT
    printf("GLOBAL\n");
    // Pointers to device memory
    FuzzyPoint *dev_centroids, *dev_data,*dev_centroidsOut, *dev_dataOut;
    //Initialise memory on device for the sizes relevant
    checkCudaErrors(cudaMalloc( (void**)&dev_centroids, NUMCLUSTER * sizeof(FuzzyPoint)));
    checkCudaErrors(cudaMalloc( (void**)&dev_data, NUMPOINTS * sizeof(FuzzyPoint)));
    checkCudaErrors(cudaMalloc( (void**)&dev_centroidsOut, NUMCLUSTER * sizeof(FuzzyPoint)));
    checkCudaErrors(cudaMalloc( (void**)&dev_dataOut, NUMPOINTS * sizeof(FuzzyPoint)));
    printf("successfully allocated memory on device \n");

    // Copy memory across
    checkCudaErrors(cudaMemcpy(dev_centroids, centroids, NUMCLUSTER * sizeof(FuzzyPoint), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_data, data, NUMPOINTS * sizeof(FuzzyPoint), cudaMemcpyHostToDevice));
    printf("successfully copied memory to device \n");

    // creating Dims
    dim3 clusterBlock(NUMCLUSTER,1,1);
    dim3 clusterThreads(1,1,1);
    // A bit uncertain about the calculation for the data
    
    printf("dim3s created \n");

    calculateCentroidsGPU<<<clusterThreads,clusterBlock>>>(dev_centroids, dev_data, dev_centroidsOut);
    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(centroids, dev_centroidsOut, NUMCLUSTER * sizeof(FuzzyPoint), cudaMemcpyDeviceToHost));
    printCentroids(centroids);
    // PARALLEL CODE SEGMENT

    // Freeing of memory
    checkCudaErrors(cudaFree(dev_centroidsOut));
    checkCudaErrors(cudaFree(dev_dataOut));
    checkCudaErrors(cudaFree(dev_centroids));
    checkCudaErrors(cudaFree(dev_data));
    checkCudaErrors(cudaDeviceReset());
    free(centroids);
    free(data);
    printf("success \n");
    return 0;
}

void testFuzzy(){
    
}