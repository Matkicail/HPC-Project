#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include "fuzzyMeans.h"
#include "../Helpers/fuzzyPoint.h"

__device__ float pointDistanceGPU(FuzzyPoint x, FuzzyPoint y){
    float sum = 0.0f;
    for(int i = 0 ; i < DIMENSIONS ; i++){
        float temp = (x.values[i] - y.values[i]);
        sum += temp * temp;
    }
    // printf("distance %f \n", sqrt(sum));
    return sqrtf(sum);
}

/**
 * Calculate the updated version of centroids based on the data points and their association
 * @param centroids the centroids that will have their values updated based on data associated with them
 * @param data the fuzzyPoints that will be used to update the centroids based on their values and their association to that centroid.
 */
__global__ void calculateCentresGPU(FuzzyPoint *data, FuzzyPoint *centroids)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int j = 0 ; j < DIMENSIONS; j++){
        float probSum = 0.0f;
        float pointSum = 0.0f; 
        for(int k = 0 ; k < NUMPOINTS ; k++){
            float temp = powf(data[k].clusters[tid],FUZZINESS);
            probSum += temp;
            pointSum += data[k].values[j]  * temp;
        }
        centroids[tid].values[j] = pointSum / probSum;
    }
}

/**
 * Calculate the updated version of centroids based on the data points and their association
 * @param centroids the centroids that will have their values updated based on data associated with them
 * @param data the fuzzyPoints that will be used to update the centroids based on their values and their association to that centroid.
 */
void calculateCentroidsOMP(FuzzyPoint *centroids, FuzzyPoint *data){
    #pragma omp parallel for
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        for(int j = 0 ; j < DIMENSIONS; j++){
            float probSum = 0.0f;
            float pointSum = 0.0f;
            #pragma omp parallel for reduction(+:probSum,pointSum)
            for(int k = 0 ; k < NUMPOINTS ; k++){
                float temp = powf(data[k].clusters[i],FUZZINESS);
                probSum += temp;
                pointSum += data[k].values[j]  * temp;
            }
            centroids[i].values[j] = pointSum / probSum;
        }
    }
}

/**
 * Calculate the updated association of data point to a centroid
 * @param data the fuzzyPoints that will have their values updated based on centroids they are associated with
 * @param centroids the centroids that will be used to update the fuzzyPoints based on their values and their association to that fuzzyPoint.
 * @param centroid the centroid we are measuring association of
 */
 __device__ double getNewValueGPU(FuzzyPoint data, FuzzyPoint centroid, FuzzyPoint *centroids){
    float p = 2.0f / (FUZZINESS-1);
    float sum = 0.0f;
    float temp;
    float distDataCentroid = pointDistanceGPU(data,centroid);
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        temp = distDataCentroid / pointDistanceGPU(data, centroids[i]);
        temp = powf(temp,p);
        sum += temp;
    }
    return 1.0f / sum;
}

/**
 * Calculate the updated association of data points based on the centroids and their association
 * @param data the fuzzyPoints that will have their values updated based on centroids they are associated with
 * @param centroids the centroids that will be used to update the fuzzyPoints based on their values and their association to that fuzzyPoint.
 */
 __global__ void updateDataAssignmentGPU(FuzzyPoint *centroids, FuzzyPoint *data){
    // For every point
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int pos = tid;
        // For point's association to that specific cluster
        for(int j = 0 ; j < NUMCLUSTER ; j++){
            float assoc = getNewValueGPU(data[pos], centroids[j], centroids);
            // printf("point: %d has %f assoc to centroid %d \n",i,assoc,j);
            data[pos].clusters[j] = assoc;
        }
}
