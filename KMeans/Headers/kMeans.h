#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "../Helpers/point.h"
#ifndef kMEANS
#define kMEANS

// basically given a malloced array init the points for clustering
#define NUMPOINTS 200000
#define ITTERATIONS 10

// A note here that should be given attention, this is not selecting a random point to turn into a centre.
// Rather it generates an entirely new point that will be considered a new point.
// This point is generated using the exact same criteria as the other points. It is just chosen as the basis of a cluster.
void initKPoints(Point *kPoints){
    for(int i = 0; i < NUMCLUSTER ; i++){
        initPoint(&kPoints[i]);
        kPoints[i].cluster = i+1;
        // printPoint(kPoints[i]);
    }
}

// not going to care too much about efficiency here given that we are going to use CUDA and MPI eventually

void averageCentroids(Point *kPoints, Point *data){
    // Create temp points that will serve as a way to store average values for a variety of pieces of info we are interested in
    // Did this to allow for easier extension
    Point *tempPoints = (Point *)malloc(NUMCLUSTER * sizeof(Point));
    // storing counts of each cluster
    int clusterCount[NUMCLUSTER];
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        clusterCount[i] = 0;
        tempPoints[i].cluster = kPoints[i].cluster;
        for(int j = 0 ; j < DIMENSIONS ; j++){
            tempPoints[i].values[j] = 0;
        }
    }
    // Now that everything is set let's find out the average

    // Collect all the points values into respective centroids
    for(int i = 0 ; i < NUMPOINTS ; i++){
        for(int j = 0 ; j < DIMENSIONS ; j++){
            tempPoints[data[i].cluster-1].values[j] += data[i].values[j];
            clusterCount[data[i].cluster-1] += 1;
        }
    }
    // average each of them now and assign this value to the original centroid
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        for(int j = 0 ; j < DIMENSIONS ; j++){
            kPoints[i].values[j] = tempPoints[i].values[j] / clusterCount[i];
        }
    }
    // free memory that is now useless
    free(tempPoints);
}

// Assigning to the closest point, so max an absolute maximal point
// This maximimal point should be vastly greater than any other possible point
// This may inhibit the program - i.e if dimensions or upper get way too large this will no longer be possible
// Using just float, maybe long float or something.
void assignDataCluster(Point *kPoints, Point *data){
    for(int i = 0 ; i < NUMPOINTS; i++){
        float distance = FLT_MAX;
        for(int j = 0 ; j < NUMCLUSTER ; j++){
            float tempDist = pointDistance(kPoints[j],data[i]);
            if(tempDist < distance){
                distance = tempDist;
                data[i].cluster = kPoints[j].cluster;
            }
        }
    }
    averageCentroids(kPoints,data);
}

void initDataPoints(Point *data){
    for(int i = 0 ; i < NUMPOINTS ; i++){
        initPoint(&data[i]);
    }
}

void printCentroids(Point *kPoints){
    int level = 1;
    printf("Printing centroids \n");
    printAllPoints(kPoints, level, NUMCLUSTER);
}

#endif