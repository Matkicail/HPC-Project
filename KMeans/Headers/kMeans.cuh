#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "kMeans.h"
#include "../Helpers/point.h"

__device__ float pointDistanceGPU(Point x, Point y)
{
     float dist = 0;
     for (int i = 0; i < DIMENSIONS; i++)
     {
         float temp = (x.values[i] - y.values[i]);
         dist += temp * temp;
     }
     return sqrtf(dist);
}

// __global__ calculateCentres(Point *data, Point *cluster){
//     // First set the general stuff to 0.
//     // I.e stuff that all clusters have in common
//     int tid = threadIdx.x;
//     float sum[DIMENSIONS];
//     int count = 0;
//     for(int i = 0 ; i < DIMENSIONS ; i++){
//         sum[i] = 0;
//     }
//     // for every point
//     for(int i = 0 ; i < NUMPOINTS ; i++){
//         // if they are in my cluster
//         if(data[i].cluster == tid + 1){
//             // increase my total count
//             count++;
//             // increase the values I hold
//             for(int j = 0 ; j < DIMENSIONS ; j++){
//                 sum[j] += data[i].values[j];
//             }
//         }
//     }
//     // Update our cluster
//     for(int i = 0 ; i < DIMENSIONS ; i++){
//         cluster->values[i] = sum[i] / count;
//     }
// }

__global__ void assignCluster(Point *data, Point *kPoints, int dataSize, int clusterSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int numAssigned = dataSize / blockDim.x;

    //32 numbers
    //16 threads

    //0,    16
    //1,    17
    //2,    18
    //3,    19
    //4,    20
    //5,    21
    //6,    22
    //7,    23
    //8,    24
    //9,    25
    //10,   26
    //11,   27
    //12,   28
    //13,   29
    //14,   30
    //15,   31


    for(int i = 0 ; i < numAssigned; i++)
    {
        //Get the position to operate on
        int pos = tid + i * dataSize;

        float distance = FLT_MAX;
        for(int j = 0 ; j < NUMCLUSTER ; j++)
        {
            float tempDist = pointDistanceGPU(kPoints[j], data[pos]);
            if(tempDist < distance)
            {
                distance = tempDist;
                data[pos].cluster = kPoints[j].cluster;
            }
        }
    }
}

