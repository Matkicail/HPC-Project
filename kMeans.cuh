#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Include the other two headers so we have access to their defines
#include "kMeans.h"
#include "point.h"


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