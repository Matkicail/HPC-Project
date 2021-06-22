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

//0-16
__global__ void calculateCentres(Point *data, Point *cluster)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float sum[DIMENSIONS] = {0};
    int count = 0;

    //For every point
    for(int i = 0; i < NUMPOINTS; i++)
    {
        //If they are in my cluster
        if(data[i].cluster == tid + 1)
        {
            //Increase my total count
            count++;
            
            //Increase the values I hold
            for(int j = 0; j < DIMENSIONS; j++)
                sum[j] += data[i].values[j];
        }
    }

    // Update our cluster
    for(int i = 0 ; i < DIMENSIONS; i++)
        if(count != 0)
            cluster[tid].values[i] = sum[i] / count;
}

//65 000
__global__ void assignCluster(Point *data, Point *kPoints, int dataSize, int clusterSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float distance = FLT_MAX;
    for(int j = 0 ; j < NUMCLUSTER ; j++)
    {
        float tempDist = pointDistanceGPU(kPoints[j], data[tid]);
        if(tempDist < distance)
        {
            distance = tempDist;
            data[tid].cluster = kPoints[j].cluster;
        }
    }
}

    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // int numAssigned = dataSize / blockDim.x;

    // for(int i = 0 ; i < numAssigned; i++)
    // {
    //     //Get the position to operate on
    //     int pos = tid + i * blockDim.x;

    //     float distance = FLT_MAX;
    //     for(int j = 0 ; j < NUMCLUSTER ; j++)
    //     {
    //         float tempDist = pointDistanceGPU(kPoints[j], data[pos]);
    //         if(tempDist < distance)
    //         {
    //             distance = tempDist;
    //             data[pos].cluster = kPoints[j].cluster;
    //         }
    //     }
    // }