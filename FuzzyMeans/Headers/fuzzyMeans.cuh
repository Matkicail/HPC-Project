#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "../Helpers/fuzzyPoint.h"

#ifndef fuzzyMEANSGPU
#define fuzzyMEANSGPU 2
#define FUZZINESS 4
#define NUMPOINTS 1000

__device__ double pointDistanceGPU(FuzzyPoint x, FuzzyPoint y)
{
    double dist = 0;
     for (int i = 0; i < DIMENSIONS; i++)
     {
        double temp = (x.values[i] - y.values[i]);
         dist += temp * temp;
     }
     return sqrtf(dist);
}

// __global__ calculateCentres(Point *data, Point *cluster){

// }

__global__ void assignClusterGPU(FuzzyPoint *data, FuzzyPoint *kPoints, int dataSize, int clusterSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int numAssigned = dataSize / blockDim.x;

}

/**
 * Calculate the updated version of centroids based on the data points and their association
 * @param centroids the centroids that will have their values updated based on data associated with them
 * @param data the fuzzyPoints that will be used to update the centroids based on their values and their association to that centroid.
 */
 void __global__ calculateCentroidsGPU(FuzzyPoint *centroids, FuzzyPoint *data){
    // Here changed i to tid
    // Will spawn as many threads as there are centroids
    // Each centroid updates itself
    // Given we probably will not use a ton of centroids, this could be done with relatively few blocks/threads
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    for(int j = 0 ; j < DIMENSIONS; j++){
        double probSum = 0.0f;
        double pointSum = 0.0f;
        for(int k = 0 ; k < NUMPOINTS ; k++){
            probSum += pow(data[k].clusters[tid],FUZZINESS);
            pointSum += data[k].values[j]  * pow(data[k].clusters[tid],FUZZINESS);
        }
        centroids[tid].values[j] = pointSum / probSum;
    }
}


#endif