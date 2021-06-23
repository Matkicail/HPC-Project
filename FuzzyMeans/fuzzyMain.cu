#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "./Helpers/fuzzyPoint.h"
#include "./Headers/fuzzyMeans.h"
#include "./Headers/fuzzyMeans.cuh"
#include "./Misc/Logger.h"
#include <omp.h>
// Includes CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check

#define NUMTHREADS 1024

void validateData(FuzzyPoint *data1, FuzzyPoint *data2, int size)
{
    for(int i = 0; i < size; i++)
    {
        for(int j = 0 ; j < NUMCLUSTER ; j++){
            if(data1[i].clusters[j] != data2[i].clusters[j])
            {
                LogError("Error: Clusters are not equal");
                // return;
            }
        }
        for(int j = 0; j < DIMENSIONS; j++)
        {
            if(data1[i].values[j] != data2[i].values[j])
            {
                LogError("Error: Values are not equal");
                printf("%f != %f\n", data1[i].values[j], data2[i].values[j]);
                // return;
            }
        }
    }

    LogPass("Test Sucessful");
}

int main()
{   
    //Create points
    FuzzyPoint *kPoints = (FuzzyPoint *)malloc(NUMCLUSTER * sizeof(FuzzyPoint));
    FuzzyPoint *kPointsTemp = (FuzzyPoint *)malloc(NUMCLUSTER * sizeof(FuzzyPoint));
    initCentroids(kPoints);
    memcpy(kPointsTemp, kPoints, NUMCLUSTER * sizeof(FuzzyPoint));

    FuzzyPoint *data = (FuzzyPoint *)malloc(NUMPOINTS * sizeof(FuzzyPoint));
    FuzzyPoint *dataTemp = (FuzzyPoint *)malloc(NUMPOINTS * sizeof(FuzzyPoint));
    initAllFuzzyPoints(data);
    memcpy(dataTemp, data, NUMPOINTS * sizeof(FuzzyPoint));

    //Serial
    double startSerial = omp_get_wtime(); 
    for(int i = 0; i < ITERATIONS; i++){
        // printCentroids(centroids);
        updateDataAssignment(kPointsTemp,dataTemp);
        calculateCentroids(kPointsTemp, dataTemp);
    }
    printf("Numpoints: %d, Numcluster: %d, Dimensions: %d \n", NUMPOINTS, NUMCLUSTER, DIMENSIONS);
    printf ("Time for the serial code: %f SECONDS\n", omp_get_wtime() - startSerial);
    // //Create memory on GPU
    // return 0;
    FuzzyPoint *deviceKPoints, *deviceData;
    cudaMalloc(&deviceKPoints, NUMCLUSTER * sizeof(FuzzyPoint));
    cudaMalloc(&deviceData, NUMPOINTS * sizeof(FuzzyPoint));

    //Copy to device
    cudaMemcpy(deviceKPoints, kPoints, NUMCLUSTER * sizeof(FuzzyPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceData, data, NUMPOINTS * sizeof(FuzzyPoint), cudaMemcpyHostToDevice);

    //Set up threads for Cluster assignment to data
    int threadsCountData = NUMTHREADS;
    int blocksCountData = (NUMPOINTS + threadsCountData - 1)  / threadsCountData;

    //Set up threads for cluster values update
    int threadsCountCluster = 1;
    int blocksCountDataCluster = NUMCLUSTER; //ASSUMING NUMCLUSTER < 1024 ALWAYS

    //Setup threads
    dim3 threadsData(threadsCountData, 1, 1);
    dim3 blocksData(blocksCountData, 1, 1);

    //Setup threads cluster
    dim3 threadsCluster(threadsCountCluster, 1, 1);
    dim3 blocksCluster(blocksCountDataCluster, 1, 1);

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    //Launch kernel
    for(int i = 0; i < ITERATIONS; i++)
    {
        updateDataAssignmentGPU<<<blocksData, threadsData>>>(deviceKPoints, deviceData);
        getLastCudaError("Kernel execution failed");
        checkCudaErrors(cudaDeviceSynchronize());
    
        calculateCentresGPU<<<blocksCluster, threadsCluster>>>(deviceData, deviceKPoints);
        getLastCudaError("Kernel execution failed");
        checkCudaErrors(cudaDeviceSynchronize());

        // updateDataAssignmentGPU<<<blocksData, threadsData>>>(deviceKPoints, deviceData);
        // getLastCudaError("Kernel execution failed");
        // checkCudaErrors(cudaDeviceSynchronize());
        // checkCudaErrors(cudaMemcpy(data, deviceData, NUMPOINTS * sizeof(FuzzyPoint), cudaMemcpyDeviceToHost));
        // calculateCentroids(kPoints, data);
        // cudaMemcpy(deviceKPoints, kPoints, NUMCLUSTER * sizeof(FuzzyPoint), cudaMemcpyHostToDevice);

    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Numpoints: %d, Numcluster: %d, Dimensions: %d \n", NUMPOINTS, NUMCLUSTER, DIMENSIONS);
    printf("Time for the kernel: %f seconds\n", time/1000);
    checkCudaErrors(cudaMemcpy(data, deviceData, NUMPOINTS * sizeof(FuzzyPoint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(kPoints, deviceKPoints, NUMCLUSTER * sizeof(FuzzyPoint), cudaMemcpyDeviceToHost));
    
    // validateData(data, dataTemp, NUMPOINTS);
    // validateData(kPoints, kPointsTemp, NUMCLUSTER);
    printf("\n####################################################################################################\n");
    cudaFree(deviceData);
    cudaFree(deviceKPoints);
    free(kPoints);
    free(data);
    return 0;
}