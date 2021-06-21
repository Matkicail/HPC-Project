#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "../Helpers/fuzzyPoint.h"

#ifndef fuzzyMEANSGPU
#define fuzzyMEANSGPU 2
#define FUZZINESS 4
#define P 4
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

/**
 * Calculate the updated association of data point to a centroid
 * @param data the fuzzyPoints that will have their values updated based on centroids they are associated with
 * @param centroids the centroids that will be used to update the fuzzyPoints based on their values and their association to that fuzzyPoint.
 * @param centroid the centroid we are measuring association of
 */
double __device__ getNewValueGPU(FuzzyPoint data, FuzzyPoint centroid, FuzzyPoint *centroids){
    double p = 2.0f / (FUZZINESS-1);
    double sum = 0.0f;
    double temp;
    double distDataCentroid = pointDistanceGPU(data,centroid);
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        temp = distDataCentroid / pointDistanceGPU(data, centroids[i]);
        temp = pow(temp,p);
        sum += temp;
    }
    return 1.0f / sum;
}

/**
 * Calculate the updated association of data points based on the centroids and their association
 * @param data the fuzzyPoints that will have their values updated based on centroids they are associated with
 * @param centroids the centroids that will be used to update the fuzzyPoints based on their values and their association to that fuzzyPoint.
 */
void __global__ updateDataAssignmentGPU(FuzzyPoint *centroids, FuzzyPoint *data, FuzzyPoint *data_out){
    // For every point
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int dataAssigned = NUMPOINTS / blockDim.x;
    // Not coalesced at all
    for(int i = 0 ; i < dataAssigned ; i++){
        // For point's association to that specific cluster
        for(int j = 0 ; j < NUMCLUSTER ; j++){
            double assoc = getNewValueGPU(data[tid + i*(dataAssigned)], centroids[j], centroids);
            // printf("point: %d has %f assoc to centroid %d \n",i,assoc,j);
            data_out[tid + i*(dataAssigned)].clusters[j] = assoc;
        }
    }
}

/**
 * Calculate the updated version of centroids based on the data points and their association
 * @param centroids the centroids that will have their values updated based on data associated with them
 * @param data the fuzzyPoints that will be used to update the centroids based on their values and their association to that centroid.
 */
 void __global__ calculateCentroidsGPU(FuzzyPoint *centroids, FuzzyPoint *data, FuzzyPoint *outCentroids){
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
        outCentroids[tid].values[j] = pointSum / probSum;
    }
}


#endif