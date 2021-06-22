#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "./Helpers/point.h"
#include "./Headers/kMeans.h"
#include "./Headers/kMeans.cuh"
#include "./Misc/Logger.h"
#include <time.h>
#include <omp.h>
// Includes CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check

#define NUMTHREADS 1024

void validateData(Point *data1, Point *data2, int size)
{
    for(int i = 0; i < size; i++)
    {
        if(data1[i].cluster != data2[i].cluster)
        {
            LogError("Error: Custers are not equal");
            break;
        }

        for(int j = 0; j < DIMENSIONS; j++)
        {
            if(data1[i].values[j] != data2[i].values[j])
            {
                LogError("Error: Values are not equal");
                printf("%f != %f\n", data1[i].values[j], data2[i].values[j]);
                break;
            }
        }
    }

    LogPass("Test Sucessful");
}

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

    double startCpu = omp_get_wtime();
 
    //Serial
    for(int i = 0; i < ITERATIONS; i++)
        assignDataCluster(kPointsTemp, dataTemp);

    printf("%f Seconds\n", omp_get_wtime() - startCpu);

    //Create memory on GPU
    Point *deviceKPoints, *deviceData;
    cudaMalloc(&deviceKPoints, NUMCLUSTER * sizeof(Point));
    cudaMalloc(&deviceData, NUMPOINTS * sizeof(Point));

    //Copy to device
    cudaMemcpy(deviceKPoints, kPoints, NUMCLUSTER * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceData, data, NUMPOINTS * sizeof(Point), cudaMemcpyHostToDevice);

    //Set up threads for Cluster assignment to data
    int threadsCountData = NUMTHREADS;
    int blocksCountData = (NUMPOINTS + threadsCountData - 1)  / threadsCountData;

    //Set up threads for cluster values update
    int threadsCountCluster = NUMCLUSTER;
    int blocksCountDataCluster = 1; //ASSUMING NUMCLUSTER < 1024 ALWAYS

    //Setup threads
    dim3 threadsData(threadsCountData, 1, 1);
    dim3 blocksData(blocksCountData, 1, 1);

    //Setup threads cluster
    dim3 threadsCluster(threadsCountCluster, 1, 1);
    dim3 blocksCluster(blocksCountDataCluster, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop, 0);

    cudaEventRecord(start);
    //Launch kernel
    for(int i = 0; i < ITERATIONS; i++)
    {
        assignCluster<<<blocksData, threadsData>>>(deviceData, deviceKPoints, NUMPOINTS, NUMCLUSTER);
        getLastCudaError("Kernel execution failed");
        checkCudaErrors(cudaDeviceSynchronize());
    
        calculateCentres<<<blocksCluster, threadsCluster>>>(deviceData, deviceKPoints);
        getLastCudaError("Kernel execution failed");
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for kernel ms: %f\n", milliseconds);
    
    checkCudaErrors(cudaMemcpy(data, deviceData, NUMPOINTS * sizeof(Point), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(kPoints, deviceKPoints, NUMCLUSTER * sizeof(Point), cudaMemcpyDeviceToHost));

    validateData(data, dataTemp, NUMPOINTS);
    validateData(kPoints, kPointsTemp, NUMCLUSTER);

    free(kPoints);
    free(data);
    return 0;
}