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

#define ITERATIONS 10

void validateCentroids(int iterations, FuzzyPoint *centroids, FuzzyPoint *data, FuzzyPoint *cudaCentroids);
void validateData(int iterations, FuzzyPoint *centroids, FuzzyPoint *data, FuzzyPoint *cudaData);

int main(){

    // SERIAL CODE SEGMENT
    // testFuzzy();
    FuzzyPoint *centroids = (FuzzyPoint *)malloc(NUMCLUSTER * sizeof(FuzzyPoint));
    FuzzyPoint *data = (FuzzyPoint *)malloc(NUMPOINTS * sizeof(FuzzyPoint));
    // Since randoming data, we need to validate that it worked
    FuzzyPoint *serialRefCentroids = (FuzzyPoint *)malloc(NUMCLUSTER * sizeof(FuzzyPoint));
    FuzzyPoint *serialRefData = (FuzzyPoint *)malloc(NUMPOINTS * sizeof(FuzzyPoint));
    initAllFuzzyPoints(data);
    // printAllFuzzyPoints(data);
    initCentroids(centroids);

    for(int i = 0 ; i < NUMCLUSTER ; i++){
        initValues(&centroids[i]);
    }
    memcpy(serialRefCentroids , centroids, NUMCLUSTER * sizeof(FuzzyPoint));
    memcpy(serialRefData , data, NUMPOINTS * sizeof(FuzzyPoint));
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
    dim3 pointThread(1024,1,1);
    dim3 pointBlock((NUMPOINTS+1023) / 1024 ,1,1);
    printf("dim3s created \n");

    calculateCentroidsGPU<<<clusterBlock,clusterThreads>>>(dev_centroids, dev_data, dev_centroidsOut);
    checkCudaErrors(cudaMemcpy(dev_centroids, dev_centroidsOut, NUMCLUSTER * sizeof(FuzzyPoint), cudaMemcpyDeviceToDevice));
    printf("calculateCentroidsGPU ran \n");
    getLastCudaError("Kernel execution failed");
    for(int i = 0 ; i < ITERATIONS ; i++){
        updateDataAssignmentGPU<<<clusterBlock,clusterThreads>>>(dev_centroids, dev_data, dev_dataOut);
        getLastCudaError("Kernel execution failed");
        checkCudaErrors(cudaMemcpy(dev_data, dev_dataOut, NUMPOINTS * sizeof(FuzzyPoint), cudaMemcpyDeviceToDevice));
        getLastCudaError("Kernel execution failed");
        calculateCentroidsGPU<<<pointBlock,pointThread>>>(dev_centroids, dev_data, dev_centroidsOut);
        getLastCudaError("Kernel execution failed");
        checkCudaErrors(cudaMemcpy(dev_centroids, dev_centroidsOut, NUMCLUSTER * sizeof(FuzzyPoint), cudaMemcpyDeviceToDevice));
        getLastCudaError("Kernel execution failed");
    }
    printf("updateDataAssignmentGPU ran \n");
    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(centroids, dev_centroidsOut, NUMCLUSTER * sizeof(FuzzyPoint), cudaMemcpyDeviceToHost));
    printf("Copied the data of dev_centroidsOut \n");
    checkCudaErrors(cudaMemcpy(data, dev_dataOut, NUMPOINTS * sizeof(FuzzyPoint), cudaMemcpyDeviceToHost));
    printf("Copied the data of dev_dataOut \n");
    printCentroids(centroids);
    // PARALLEL CODE SEGMENT
    
    // Validating that serial and parallel produce the same
    validateData(1, serialRefCentroids, serialRefData,data);
    validateCentroids(1, serialRefCentroids, data, centroids);
    // Validating that serial and parallel produce the same

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

void validateCentroids(int iterations, FuzzyPoint *centroids, FuzzyPoint *data, FuzzyPoint *cudaCentroids){
    for(int i = 0 ; i < iterations ; i++){
        calculateCentroids(centroids, data);
    }
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        for(int j = 0 ; j < DIMENSIONS ; j++){
            if(centroids[i].values[j] != cudaCentroids[i].values[j]){
                if(fabs(centroids[i].values[j] - cudaCentroids[i].values[j]) > EPSILON){
                    printf("An error occured at i: %d j: %d\n",i,j);
                    printf("centroidVal = %f, cudaCentroidVal = %f\n", centroids[i].values[j], cudaCentroids[i].values[j]);
                    j = DIMENSIONS;
                    i = NUMCLUSTER;
                }
            }
        }
    }
}

void validateData(int iterations, FuzzyPoint *centroids, FuzzyPoint *data, FuzzyPoint *cudaData){
    for(int i = 0 ; i < iterations ; i++){
        calculateCentroids(centroids, data);
        updateDataAssignment(centroids, data);
    }
    for(int i = 0 ; i < NUMPOINTS ; i++){
        for(int j = 0 ; j < DIMENSIONS ; j++){
            if(data[i].values[j] != cudaData[i].values[j] || data[i].clusters[j] != cudaData[i].clusters[j]){
                if(fabs(data[i].values[j] != cudaData[i].values[j]) > EPSILON){
                    printf("An error occured at i: %d j: %d\n",i,j);
                    printf("dataVal = %f, cudaDataVal = %f\n", data[i].values[j], cudaData[i].values[j]);
                    j = DIMENSIONS;
                    i = NUMCLUSTER;
                }
                if( fabs(data[i].clusters[j] != cudaData[i].clusters[j]) > EPSILON){
                    printf("An error occured at i: %d j: %d\n",i,j);
                    printf("dataCluster = %f, cudaDataCluster = %f\n", data[i].clusters[j], cudaData[i].clusters[j]);
                    j = DIMENSIONS;
                    i = NUMCLUSTER;
                }
            }
        }
    }
}