#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Include the other two headers so we have access to their defines
#include "kMeans.h"
#include "point.h"

// Idea here is to slosh betweeen the two with the following flow
// send data
// assign centres
// calculate the centres new mean
// receive data
// cylce for ITTERATIONS
// This would require two different dims for the blocks and threads respectively
// One which matches to the calculateCentres which requires us to only look at the K centres
// One which matches to the assignCluster which requires us to look at M points where M is vastly less than N

// assume we give each cluster a thread
// Note that the below would need the cluster in memory to conistently be the updated cluster
// It would also require the data to be up to date given that we need it to be in the right cluster
__global__ calculateCentres(Point *data, Point *cluster){
    // First set the general stuff to 0.
    // I.e stuff that all clusters have in common
    int tid = threadIdx.x;
    float sum[DIMENSIONS];
    int count = 0;
    for(int i = 0 ; i < DIMENSIONS ; i++){
        sum[i] = 0;
    }
    // for every point
    for(int i = 0 ; i < NUMPOINTS ; i++){
        // if they are in my cluster
        if(data[i].cluster == tid + 1){
            // increase my total count
            count++;
            // increase the values I hold
            for(int j = 0 ; j < DIMENSIONS ; j++){
                sum[j] += data[i].values[j];
            }
        }
    }
    // Update our cluster
    for(int i = 0 ; i < DIMENSIONS ; i++){
        cluster->values[i] = sum[i] / count;
    }
    __syncthreads();
}

// Need to discuss this one with you.
// Idea is either to span N data points based on maximum number of blocks/threads
// This doesnt work really since we could do vastly more, so idea could be to assign M data points to each thread
// That thread figures out the assignment for those M data points
// Currently not coallesced and unsure if it will work, it relies on the serial implementation but instead with a num assigned and tid
__global__ assignCluster(Point *data, Point *clusters, int numAssigned){
    int tid = blockIdx.x*blockDim.x threadIdx.x;
    for(int i = 0 ; i < numAssigned; i++){
        if(i + tid < NUMPOINTS){
            float distance = pow(UPPER,DIMENSIONS) + UPPER;
            for(int j = 0 ; j < NUMCLUSTER ; j++){
                float tempDist = pointDistance(kPoints[j],data[tid + i]);
                if(tempDist < distance){
                    distance = tempDist;
                    data[tid + i].cluster = kPoints[j].cluster;
                }
            }
        }
    }
    __syncthreads();
}

