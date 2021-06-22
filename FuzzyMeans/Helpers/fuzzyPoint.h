#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#ifndef FUZZYPOINT
#define FUZZYPOINT

#define DIMENSIONS (1 << 1)
#define NUMCLUSTER (1 << 6)
#define UPPER 1000000
#define LOWER 0
#define EPSILON 0.05

// Make this a class
typedef struct FuzzyPoints{
    // Chose to make this a double, we can change it to a double but don't think it is too important.
    float values[DIMENSIONS];
    // Need to represent the probability that this is associated with a specific cluster
    float clusters[NUMCLUSTER];
} FuzzyPoint;
/**
 * Initialize a point's probabilities to random. See fuzzyMean.h to initialize data or clusters with those helper functions.
 * @param x Point that needs to be initialized with probabilities
 */
void initProb(FuzzyPoint *x){

    for(int i = 0 ; i < NUMCLUSTER ; i++){
        x->clusters[i] = 0.0f;
    }

    bool generating = true;
    int i = 0;
    float tempProb = 0.0f;
    while(generating){
        // The prime number chosen can be improved
        // But this would need to be done empirically
        float prob = rand() % 200;
        prob /= 1000;
        if(tempProb + prob > 1){
            x->clusters[i] += 1-tempProb;
            generating = false;
        }
        else{
            tempProb += prob;
            x->clusters[i] += prob;
        }

        i++;
        if(i == NUMCLUSTER){
            i = 0;
        }
    }

}
/**
 * Initialize a point's values to random. See fuzzyMean.h to initialize data or clusters with those helper functions.
 * @param x Point that needs to be initialized with values in each dimension
 */
void initValues(FuzzyPoint *x){
    int myCluster = (rand() % (NUMCLUSTER - 1 + 1)) + 1;
    int localLower = UPPER / NUMCLUSTER;
    int range = UPPER;
    for(int i = 0 ; i < DIMENSIONS ; i++){
        x->values[i] = (rand() % (localLower*(myCluster+1) - localLower*myCluster + 1)) + localLower*myCluster;
    }
}
/**
 * Lowest level to initialize a data point to random. See fuzzyMean.h to initialize data or clusters with those helper functions.
 * @param x Point that needs to be initialized
 */
void initFuzzyPoint(FuzzyPoint *x){
    initProb(x);
    initValues(x);
}
/**
 * Initialize centroids to random. See fuzzyMean.h to initialize data or clusters with those helper functions.
 * @param x Centroids that need to be initialized
 */
void initCentroids(FuzzyPoint *centroid){
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        for(int j = 0 ; j < DIMENSIONS ; j++){
            centroid[i].values[j] = 0;
        }
    }
    
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        for(int j = 0 ; j < NUMCLUSTER ; j++){
            if(i == j){
                centroid[i].clusters[j]=1;
            }
            else{
                centroid[i].clusters[j]=0;
            }
        }
    }
}
/**
 * Just a method to check if the probability of a fuzzyPoint adds to 1
 * @param x Point that nis being checked
 */
void testProbability(FuzzyPoint x){
    double tempSum = 0.0f;
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        tempSum += x.clusters[i];
        if(tempSum > 1.0f){
            printf("failed \n");
        }
    }
}

/**
 * Calculate the distance between to fuzzyPoints using L2 norm
 * @param x a fuzzyPoint
 * @param y a fuzzyPoint
 */
double distance(FuzzyPoint x, FuzzyPoint y){
    float sum = 0.0f;
    for(int i = 0 ; i < DIMENSIONS ; i++){
        float temp = (x.values[i] - y.values[i]);
        sum += temp * temp;
    }
    // printf("distance %f \n", sqrt(sum));
    return sqrtf(sum);
}

/**
 * Function to print a number of tabs, used in general printing ouf output
 * @param level number of indents
 */
void printtabs(int level){
    for(int i = 0 ; i < level ; i++){
        printf("\t");
    }
}
/**
 * Prints a single fuzzyPoint and can be used with a centroid or with a data point.
 * @param point Point that will have its cluster associations & values printed.
 */
void printFuzzyPoint(FuzzyPoint point){
    printf("Cluster Assoc: ");
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        printf("%f ", point.clusters[i]);
    }
    printf("\n");
    printtabs(1);
    printf("Cluster Values: ");
    for(int i = 0 ; i < DIMENSIONS; i++){
        printf("%f ", point.values[i]);
    }
    printf("\n");
}

#endif