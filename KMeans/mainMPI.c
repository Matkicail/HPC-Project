#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "./Helpers/point.h"
#include "./Headers/kMeans.h"

float pointDistanceMPI(Point x, Point y)
{
     float dist = 0;
     for (int i = 0; i < DIMENSIONS; i++)
     {
         float temp = (x.values[i] - y.values[i]);
         dist += temp * temp;
     }
     return sqrtf(dist);
}

void calculateCentresMPI(Point *data, Point *kPoints, int pointLength)
{
    //Temporary cluster points to use for averaging
    Point *tempPoints = (Point *)malloc(NUMCLUSTER * sizeof(Point));

    //Storing counts of each cluster to average
    for (int i = 0; i < NUMCLUSTER; i++)
    {
        tempPoints[i].cluster = kPoints[i].cluster;
        for (int j = 0; j < DIMENSIONS; j++)
            tempPoints[i].values[j] = 0;
    }

    //Calculate the sum of the points to get the average
    for (int i = 0; i < pointLength; i++)
    {
        for (int j = 0; j < DIMENSIONS; j++)
            tempPoints[data[i].cluster - 1].values[j] += data[i].values[j];

        tempPoints[data[i].cluster - 1].clusterCount += 1;
    }

    //Average each of them now and assign this value to the original centroid
    for (int i = 0; i < NUMCLUSTER; i++)
    {
        for (int j = 0; j < DIMENSIONS; j++)
			kPoints[i].values[j] = tempPoints[i].values[j];

		kPoints[i].clusterCount = tempPoints[i].clusterCount;
    }

    //Free memory that is now useless
    free(tempPoints);
}

void assignClusterMPI(Point *data, Point *cluster, int pointLength)
{
	for(int i = 0; i < pointLength; i++)
	{
        float distance = FLT_MAX;
		for(int j = 0; j < NUMCLUSTER; j++)
		{
			float tempDist = pointDistance(cluster[j], data[i]);
            if (tempDist < distance)
            {
                distance = tempDist;
                data[i].cluster = cluster[j].cluster;
            }
		}
	}
}

int main(int argc, char *argv[])
{
	int nproces, myrank;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproces);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	Point *kPoints, *data, *dataTemp, *tempKPoints;


	int pointLength = (NUMPOINTS/nproces);
	kPoints = (Point *)malloc(NUMCLUSTER * sizeof(Point));
	tempKPoints = (Point *)malloc(NUMCLUSTER * sizeof(Point));
	initDataPointsToValue(tempKPoints, NUMCLUSTER, 0);


	//If master thread, create all points
	if(myrank == 0)
	{
		initKPoints(kPoints);
		dataTemp = (Point *)malloc(NUMPOINTS * sizeof(Point));
	    initDataPoints(dataTemp);
	}

	data = (Point *)malloc(pointLength * sizeof(Point));
	initDataPointsToValue(data, pointLength, 0);

	//Broadcase clusters and scatter points
	MPI_Bcast(kPoints, NUMCLUSTER * sizeof(Point), MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(dataTemp, pointLength * sizeof(Point), MPI_CHAR, data, pointLength * sizeof(Point), MPI_CHAR, 0, MPI_COMM_WORLD);

	for(int i = 0; i < 1; i++)
	{
		assignClusterMPI(data, kPoints, pointLength);
		calculateCentresMPI(data, kPoints, pointLength);

		//Send and receive
		if(myrank == 0)
		{
			for(int j = 1; j < nproces; j++)
			{
				MPI_Recv(tempKPoints, NUMCLUSTER * sizeof(Point), MPI_CHAR, j, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				//Sum over the arrived values
				for(int k = 0; k < NUMCLUSTER; k++)
				{
					for(int m = 0; m < DIMENSIONS; m++)
					{
						kPoints[k].values[m] += tempKPoints[k].values[m];
						kPoints[k].clusterCount += tempKPoints[k].clusterCount;
					}
				}
			}
		}
		else
			MPI_Send(kPoints, NUMCLUSTER * sizeof(Point), MPI_CHAR, 0, 20, MPI_COMM_WORLD);

		//Average results
		for(int k = 0; k < NUMCLUSTER; k++)
			for(int m = 0; m < DIMENSIONS; m++)
				kPoints[k].values[m] /= kPoints[k].clusterCount;

		//Send updated values
		MPI_Bcast(kPoints, NUMCLUSTER * sizeof(Point), MPI_CHAR, 0, MPI_COMM_WORLD);
	}

	if (myrank == 0)
	{
		printCentroids(kPoints);
	}

    MPI_Finalize();
    return 0;
}