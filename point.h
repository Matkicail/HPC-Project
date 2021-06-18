#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#ifndef POINT
#define POINT
// compiler directives
#define DIMENSIONS 3
#define NUMCLUSTER 8
#define UPPER 20
#define LOWER -20

// make this a class
typedef struct Points{
    // Chose to make this a float, we can change it to a double but don't think it is too important.
    float values[DIMENSIONS];
    int cluster;
} Point;
// assign to be a random values in D dims
void randomValues(Point *x){
    for(int i = 0 ; i < DIMENSIONS ; i++){
        x->values[i] = (rand() % (UPPER - LOWER + 1)) + LOWER;
    }
}
// change the characteristics 
void initPoint(Point *x){
    x->cluster = -1;
    randomValues(x);
}
// print the point
void printPoint(Point x){
    printf("Cluster %d Values:",x.cluster);
    for(int i = 0 ; i < DIMENSIONS ; i++){
        if(i != DIMENSIONS -1){
            printf(" %f,",x.values[i]);
        }
        else{
            printf(" %f",x.values[i]);
        }
    }
    printf("\n");
}
// making a function to handle changing cluster assignment
void assignCluster(Point *x, int cluster){
    x->cluster = cluster;
}

// Here I chose to use Minkowski distance for maximum flex
// Basically it is like Euclidean mixed with manhattan
// https://en.wikipedia.org/wiki/Minkowski_distance
float pointDistance(Point x, Point y){
    float dist = 0;
    for(int i = 0; i < DIMENSIONS; i++){
        dist += (x.values[i] - y.values[i])*(x.values[i] - y.values[i]);
    }
    return pow(dist, 1.0/DIMENSIONS);
}

#endif