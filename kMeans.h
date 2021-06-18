#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "point.h"

#ifndef kMEANS
#define kMEANS
// basically given a malloced array init the points for clustering
void initKPoints(Point *kPoints){
    for(int i = 0; i < NUMCLUSTER ; i++){
        initPoint(&kPoints[i]);
        kPoints[i].cluster = i+1;
        // printPoint(kPoints[i]);
    }
}

#endif