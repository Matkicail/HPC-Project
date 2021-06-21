#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "./Helpers/point.h"
#include "./Headers/kMeans.h"
#include "./Headers/kMeans.cuh"

// Includes CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check

#define NUMTHREADS 16

int main()
{
    //Create points
    Point *kPoints = (Point *)malloc(NUMCLUSTER * sizeof(Point));
    Point *kPointsTemp = (Point *)malloc(NUMCLUSTER * sizeof(Point));
    initKPoints(kPoints);
    memcpy(kPointsTemp, kPoints, NUMCLUSTER * sizeof(Point));

    Point *data = (Point *)malloc(NUMPOINTS * sizeof(Point));
    Point *dataTemp = (Point *)malloc(NUMPOINTS * sizeof(Point));
    initDataPoints(data);
    memcpy(dataTemp, data, NUMPOINTS * sizeof(Point));

    //Serial
    assignDataCluster(kPointsTemp, dataTemp);
    printDataPoints(dataTemp, NUMPOINTS);
    printf("\n\n");

    //Create memory on GPU
    Point *deviceKPoints, *deviceData;
    cudaMalloc(&deviceKPoints, NUMCLUSTER * sizeof(Point));
    cudaMalloc(&deviceData, NUMPOINTS * sizeof(Point));

    //Copy to device
    cudaMemcpy(deviceKPoints, kPoints, NUMCLUSTER * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceData, data, NUMPOINTS * sizeof(Point), cudaMemcpyHostToDevice);


    //Set up threads
    int threadsCount = NUMTHREADS;
    int blocksCount = (NUMPOINTS + threadsCount - 1)  / threadsCount;

    //Setup threads
    dim3 threads(threadsCount, 1, 1);
    dim3 blocks(blocksCount, 1, 1);

    //Launch kernel
    assignCluster<<<blocks, threads>>>(deviceData, deviceKPoints, NUMPOINTS, NUMCLUSTER);
    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());
    cudaMemcpy(data, deviceData, NUMPOINTS * sizeof(Point), cudaMemcpyDeviceToHost);

    printDataPoints(data, NUMPOINTS);

    free(kPoints);
    free(data);
    return 0;
}