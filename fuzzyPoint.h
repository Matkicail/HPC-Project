#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#ifndef FUZZYPOINT
#define FUZZYPOINT

#define DIMENSIONS 2
#define NUMCLUSTER 7
#define UPPER 10000
#define LOWER -10000
#define epsilon 0.0001

// Fuzzy means clustering
typedef struct FuzzyPoints{
    // Chose to make this a float, we can change it to a double but don't think it is too important.
    float values[DIMENSIONS];
    // Need to represent the probability that this is associated with a specific cluster
    float cluster[NUMCLUSTER];
} FuzzyPoint;

// assign to be a random values in D dims
void randomValues(FuzzyPoint *x){
    for(int i = 0 ; i < DIMENSIONS ; i++){
        x->values[i] = (rand() % (UPPER - LOWER + 1)) + LOWER;
    }
}
// In order to simulate random probability
// I will make use of a prime number with a mod class
// I can explain this later, but I chose this instead of doing uniform probability
// If we can use uniform probability, that would be better as it is simplier and less computationally involved
void randomProbabilities(float *clusters){
    for(int i = 0 ; i < NUMCLUSTER; i++){
        clusters[i] = 0;
    }
    // the below just attempts to create the probabilities
    // in a round robin fashion where it adds to the probabilities of clusters
    // as long as the total probability assigned is < 1
    // when the probability > 1 we find out what the last probability should be based on 1 - sum of rest of probabilities of clusters at current
    int i = 0;
    float temp = 0.0f;
    bool generating = true;
    while(generating){
        float val = (rand() % (13 - 0 + 0)) + 0;
        val = val / 100;
        temp += val;
        if(temp <= 1.0f){
            clusters[i] += val;
            if(temp == 1.0f){
                return;
            }
        }
        else{
            generating = false;
            float totProb = 0.0f;
            for(int j = 0 ; j < NUMCLUSTER ; j++){
                totProb += clusters[j];
            }
            clusters[i] = 1 - totProb;
            
        }
        if(i < NUMCLUSTER){
            i++;
        }
        else{
            i = 0;
        }
    }
    return;
}
// change the characteristics 
void initPoint(FuzzyPoint *x){
    randomProbabilities(x->cluster);
    randomValues(x);
}

void initFuzzyPoints(FuzzyPoint *data, int num){
    for(int i = 0 ; i < num ; i++){
        initPoint(&data[i]);
    }
}

// I just like the printtabs function as output looks better
void printtabs(int level){
    for(int i = 0 ; i < level ; i++){
        printf("\t");
    }
}

// Just a basic function to init centroids
// basically set everything to do with them to 0
void initCentroids(FuzzyPoint *centroids){
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        for(int j = 0 ; j < DIMENSIONS ; j++){
            centroids[i].values[j] = 0;
            centroids[i].cluster[j] = 0;
        }
    }
}
// Currently not working
void calculateCentroids(FuzzyPoint *centroids, FuzzyPoint *data, int numPoints){
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        for(int j = 0 ; j < DIMENSIONS; j++){
            float tempProb = 0.0f;
            float tempVal = 0.0f;
            for(int k = 0 ; k < numPoints ; k++){
                float prob = data[k].cluster[j];
                tempProb += prob;
                tempVal += prob * data[k].values[j];
            }
            centroids[i].values[j] = tempVal / tempProb;
        }
    }
}

// changed the print function so that it can show the cluster assignment
void printPoint(FuzzyPoint x){
    printf("Cluster Values: ");
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        if(i != NUMCLUSTER -1){
            printf(" %f,",x.cluster[i]);
        }
        else{
            printf(" %f",x.cluster[i]);
        }
    }
    printf("\n");
    printtabs(1);
    printf("Values:");
    
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

void printFuzzyPoints(FuzzyPoint *x, int num){
    for(int i = 0 ; i < num ; i++){
        printPoint(x[i]);
    }
}



#endif