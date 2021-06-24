#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "../Helpers/point.h"
#ifndef kMEANS
#define kMEANS

// basically given a malloced array init the points for clustering
#define NUMPOINTS (1<<18)
#define ITERATIONS 10

/**
 * Calculate the average centroid
 * @param kPoints array of clusters
 * @param data array of data points
 */
void averageCentroids(Point *kPoints, Point *data)
{
    //Temporary cluster points to use for averaging
    Point *tempPoints = (Point *)malloc(NUMCLUSTER * sizeof(Point));

    //Storing counts of each cluster to average
    int clusterCount[NUMCLUSTER] = {0};
    for (int i = 0; i < NUMCLUSTER; i++)
    {
        tempPoints[i].cluster = kPoints[i].cluster;
        for (int j = 0; j < DIMENSIONS; j++)
            tempPoints[i].values[j] = 0;
    }

    //Calculate the sum of the points to get the average
    for (int i = 0; i < NUMPOINTS; i++)
    {
        for (int j = 0; j < DIMENSIONS; j++)
        {
            tempPoints[data[i].cluster - 1].values[j] += data[i].values[j];
        }
        clusterCount[data[i].cluster - 1] += 1;
    }

    //Average each of them now and assign this value to the original centroid
    for (int i = 0; i < NUMCLUSTER; i++)
    {
        for (int j = 0; j < DIMENSIONS; j++)
        {
            if(clusterCount[i] != 0)
                kPoints[i].values[j] = tempPoints[i].values[j] / clusterCount[i];
        }
    }

    //Free memory that is now useless
    free(tempPoints);
}

/**
 *  Assigns the datapoint to a cluster and average cluster
 *  @param kPoints cluster points
 *  @param data points to be assigned to a cluster
 */
void assignDataCluster(Point *kPoints, Point *data)
{
    for (int i = 0; i < NUMPOINTS; i++)
    {
        float distance = FLT_MAX;
        for (int j = 0; j < NUMCLUSTER; j++)
        {
            float tempDist = pointDistance(kPoints[j], data[i]);
            if (tempDist < distance)
            {
                distance = tempDist;
                data[i].cluster = kPoints[j].cluster;
            }
        }
    }

    //Average cluster
    averageCentroids(kPoints, data);
}

/**
 * For initializing all datapoints, similar to initDataPoints but this also sets the cluster assignment
 * @param data array of points to be initialized
 */
void initKPoints(Point *kPoints)
{
    for (int i = 0; i < NUMCLUSTER; i++)
    {
        initPoint(&kPoints[i]);
        kPoints[i].cluster = i + 1;
    }
}

/**
 * For initializing all datapoints, same as initKPoints but doesn't set the cluster assignment
 * @param data array of points to be initialized
 */
void initDataPoints(Point *data)
{
    for (int i = 0; i < NUMPOINTS; i++)
        initPoint(&data[i]);
}

/**
 * For initializing all datapoints, same as initKPoints but doesn't set the cluster assignment
 * @param data array of points to be initialized
 */
void initDataPointsToValue(Point *data, int dataSize, float value)
{
    for (int i = 0; i < dataSize; i++)
    {
        initPoint(&data[i]);
        for (int j = 0; j < DIMENSIONS; j++)
            data[i].values[j] = value;
    }
}

/**
 * Helper function to print centroids
 * @param kPoints An array of centroids
 */
void printCentroids(Point *kPoints)
{
    int level = 1;
    printf("Printing centroids \n");
    printAllPoints(kPoints, level, NUMCLUSTER);
}

#endif