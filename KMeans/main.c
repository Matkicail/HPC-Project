#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "./Helpers/point.h"
#include "./Headers/kMeans.h"

void testPoints();

int main(){
    
    Point *kPoints = (Point *)malloc(NUMCLUSTER * sizeof(Point));
    initKPoints(kPoints);
    printCentroids(kPoints);
    Point *data = (Point *)malloc(NUMPOINTS * sizeof(Point));
    initDataPoints(data);
    for(int i = 0 ; i < ITTERATIONS ; i++){
        assignDataCluster(kPoints, data);
    }
    printCentroids(kPoints);
    // printDataPoints(data, NUMPOINTS);


    free(kPoints);
    free(data);
    return 0;
}

// just a simple test I made to see if the points init
void testPoints(){
    Point x,y;
    initPoint(&y);
    initPoint(&x);
    printPoint(y);
    printPoint(x);
    printf("Distance %f \n",pointDistance(x,y));
    return;
}
