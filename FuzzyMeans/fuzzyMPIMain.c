#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <mpi.h>
#include "./Misc/Logger.h"
#include "./Headers/fuzzyMeans.h"


#define RECEIVE 0
#define SEND 1

void validateData(FuzzyPoint *data1, FuzzyPoint *data2, int size, float epsilon)
{
    for(int i = 0; i < size; i++)
    {
        for(int j = 0 ; j < NUMCLUSTER ; j++){
            if(fabsf(data1[i].clusters[j] - data2[i].clusters[j]) > epsilon)
            {
                LogError("Error: Clusters are not equal");
                return;
            }
        }
        for(int j = 0; j < DIMENSIONS; j++)
        {
            if(fabsf(data1[i].values[j] - data2[i].values[j]) > epsilon)
            {
                LogError("Error: Values are not equal");
                printf("%f != %f\n", data1[i].values[j], data2[i].values[j]);
                return;
            }
        }
    }

    LogPass("Test Sucessful");
}

void calculateCentresMPI(FuzzyPoint *data, FuzzyPoint *centroids, int pointLength)
{
    
    //Average each of them now and assign this value to the original centroid
    for(int i = 0 ; i < NUMCLUSTER ; i++){
        for(int j = 0 ; j < DIMENSIONS; j++){
            float probSum = 0.0f;
            float pointSum = 0.0f;
            for(int k = 0 ; k < pointLength ; k++){
                probSum += powf(data[k].clusters[i],FUZZINESS);
                pointSum += data[k].values[j]  * powf(data[k].clusters[i],FUZZINESS);
            }
            centroids[i].clusterProbSum[j] = probSum;
            centroids[i].clusterProbSum[j] = pointSum;
        }
    }
}

void updateDataAssignmentMPI(FuzzyPoint *data, FuzzyPoint *centroids, int pointLength)
{
	for(int i = 0; i < pointLength; i++)
	{
		for(int j = 0; j < NUMCLUSTER; j++)
		{
            float assoc = getNewValue(data[i], centroids[j], centroids);
            data[i].clusters[j] = assoc;
		}
	}
}

int main(int argc, char *argv[]){
    
    // Creating MPI processes
    int nproces, myrank;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproces);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	FuzzyPoint *kPoints, *data, *dataTemp, *tempKPoints, *serialKPoints, *serialData;

	int pointLength = (NUMPOINTS/nproces);
	kPoints = (FuzzyPoint *)malloc(NUMCLUSTER * sizeof(FuzzyPoint));
	tempKPoints = (FuzzyPoint *)malloc(NUMCLUSTER * sizeof(FuzzyPoint));
	// initDataPointsToValue(tempKPoints, NUMCLUSTER, 0);

	//If master thread, create all points
	if(myrank == 0)
	{
		//For serial validation
		serialKPoints = (FuzzyPoint *)malloc(NUMCLUSTER * sizeof(FuzzyPoint));
		serialData = (FuzzyPoint *)malloc(NUMPOINTS * sizeof(FuzzyPoint));

		dataTemp = (FuzzyPoint *)malloc(NUMPOINTS * sizeof(FuzzyPoint));
		initCentroids(kPoints);
	    initAllFuzzyPoints(dataTemp);

    	memcpy(serialKPoints, kPoints, NUMCLUSTER * sizeof(FuzzyPoint));
    	memcpy(serialData, dataTemp, NUMPOINTS * sizeof(FuzzyPoint));
	}

	data = (FuzzyPoint *)malloc(pointLength * sizeof(FuzzyPoint));
	// initDataPointsToValue(data, pointLength, 0);

	//Broadcase clusters and scatter points
	MPI_Bcast(kPoints, NUMCLUSTER * sizeof(FuzzyPoint), MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(dataTemp, pointLength * sizeof(FuzzyPoint), MPI_CHAR, data, pointLength * sizeof(FuzzyPoint), MPI_CHAR, 0, MPI_COMM_WORLD);

	for(int i = 0; i < 1; i++)
	{
		updateDataAssignmentMPI(data, kPoints, pointLength);
		calculateCentresMPI(data, kPoints, pointLength);

		//Send and receive
		if(myrank == 0)
		{
			for(int j = 1; j < nproces; j++)
			{
				MPI_Recv(tempKPoints, NUMCLUSTER * sizeof(FuzzyPoint), MPI_CHAR, j, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				//Sum over the arrived values
				for(int k = 0; k < NUMCLUSTER; k++)
				{
                    float totProbSum = 0.0f;
                    float totPointSum = 0.0f;
					for(int m = 0; m < DIMENSIONS; m++){
						kPoints[k].clusterProbSum[m] += tempKPoints[k].clusterProbSum[m];
                        kPoints[k].clusterPointSum[m] += tempKPoints[k].clusterPointSum[m];
                    }
				}
			}
		}
		else
			MPI_Send(kPoints, NUMCLUSTER * sizeof(FuzzyPoint), MPI_CHAR, 0, 20, MPI_COMM_WORLD);

		//Average results
		for(int k = 0; k < NUMCLUSTER; k++)
			for(int m = 0; m < DIMENSIONS; m++)
				kPoints[k].values[m] = kPoints[k].clusterPointSum[m]/kPoints[k].clusterProbSum[m];

		//Send updated values
		MPI_Bcast(kPoints, NUMCLUSTER * sizeof(FuzzyPoint), MPI_CHAR, 0, MPI_COMM_WORLD);
	}

	if(myrank == 0)
	{
		updateDataAssignment(serialKPoints, serialData);
    	validateData(kPoints, serialKPoints, NUMCLUSTER, 1);
	}


    MPI_Finalize();
    return 0;

}