#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#ifndef FUZZYPOINT
#define FUZZYPOINT

// Please note an issue here that has to be considered important
// Dimensions must be <= number of clusters. If this is not the case then things will break
#define DIMENSIONS 2
#define NUMCLUSTER 10
#define UPPER 10000
#define LOWER -10000
#define epsilon 0.0001

// Fuzzy means clustering
typedef struct FuzzyPoints{
    // Chose to make this a float, we can change it to a double but don't think it is too important.
    float values[DIMENSIONS];
    // Need to represent the probability that this is associated with a specific cluster
    float clusters[NUMCLUSTER];
} FuzzyPoint;

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
        float prob = rand() % 21;
        prob /= 100;
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
void initValues(FuzzyPoint *x){
    for(int i = 0 ; i < DIMENSIONS ; i++){
        x->values[i] = (rand() % (UPPER - LOWER + 1)) + LOWER;
    }
}

void initFuzzyPoint(FuzzyPoint *x){
    initProb(x);
    initValues(x);
}

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

void testProbability(FuzzyPoint x){
    float tempSum = 0.0f;
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        tempSum += x.clusters[i];
        if(tempSum > 1.0f){
            printf("failed \n");
        }
    }
}

void printtabs(int level){
    for(int i = 0 ; i < level ; i++){
        printf("\t");
    }
}

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