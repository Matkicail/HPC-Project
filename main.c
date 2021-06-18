#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "point.h"
#include "kMeans.h"

void testPoints();

int main(){
    
    Point *kPoints = (Point *)malloc(NUMCLUSTER * sizeof(Point));
    initKPoints(kPoints);
    
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
