#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#ifndef POINT
#define POINT

// compiler directives
#define DIMENSIONS 12
#define NUMCLUSTER 60
#define UPPER 10000
#define LOWER -10000

// make this a class
typedef struct Points{
    // Chose to make this a float, we can change it to a double but don't think it is too important.
    float values[DIMENSIONS];
    int cluster;
} Point;

void printtabs(int level)
{
    for (int i = 0; i < level; i++)
        printf("\t");
}

// assign to be a random values in D dims
void randomValues(Point *x)
{
    for (int i = 0; i < DIMENSIONS; i++)
        x->values[i] = (rand() % (UPPER - LOWER + 1)) + LOWER;
}

/**
 * Lowest level to initialize a point to random. See kMeans.h to initialize data or clusters with those helper functions.
 * @param x Point that needs to be initialized
 */
void initPoint(Point *x)
{
    x->cluster = -1;
    randomValues(x);
}

// print the point
void printPoint(Point x)
{
    printf("Cluster %d Values:", x.cluster);
    for (int i = 0; i < DIMENSIONS; i++)
    {
        if (i != DIMENSIONS - 1)
            printf(" %f,", x.values[i]);
        else
            printf(" %f", x.values[i]);
    }
    printf("\n");
}

// making a function to handle changing cluster assignment
void assignCluster(Point *x, int cluster)
{
    x->cluster = cluster;
}

/**
 * The Euclidean distance metric
 * @param x point x
 * @param y point y
 * returns the distance from point x and y
 */
float pointDistance(Point x, Point y)
{
    float dist = 0;
    for (int i = 0; i < DIMENSIONS; i++)
    {
        float temp = (x.values[i] - y.values[i]);
        dist += temp * temp;
    }
    return pow(dist, 1.0 / 2);
}

/**
* Mainly just call this when you want to print points for some structure or system.
* @param data points to print
* @param level number of tabs
* @param num numer to loop pver
*/ 
void printAllPoints(Point *data, int level, int num)
{
    for (int i = 0; i < num; i++)
    {
        printtabs(level);
        printPoint(data[i]);
    }
}

/**
* Helper function to call printAllPoints. Level is set to 2 by default which is passed to printAllPoints
* @param data points to print
* @param level number of tabs
* @param num numer to loop pver
*/ 
void printDataPoints(Point *data, int num)
{
    int level = 2;
    printf("Printing data points \n");
    printAllPoints(data, level, num);
}

#endif