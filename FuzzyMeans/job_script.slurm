#!/bin/bash
# specify a partition
#SBATCH -p batch
# specify number of nodes
#SBATCH -N 16
# specify number of cores
#SBATCH -n 128
# specify memory pool for all cores
##SBATCH --mem 10
# specify the wall clock time limit for the job
#SBATCH -t 09:00:00
# specify the job name
#SBATCH -J FuzzyClustering
# specify the filename to be used for writing output
#SBATCH -o /home-mscluster/mkruger/openMPI1669326.%N.%j.out
# specify the filename for stderr
#SBATCH -e /home-mscluster/mkruger/openMPI1669326.%N.%j.err


echo ------------------------------------------------------
echo -n 'Job is running on node ' $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SLURM: sbatch is running on $SLURM_SUBMIT_HOST
echo SLURM: job ID is $SLURM_JOB_ID
echo SLURM: submit directory is $SLURM_SUBMIT_DIR
echo SLURM: number of nodes allocated is $SLURM_JOB_NUM_NODES
echo SLURM: number of cores is $SLURM_NTASKS
echo SLURM: job name is $SLURM_JOB_NAME
echo ------------------------------------------------------
# cd $SLURM_SUBMIT_DIR
echo 'Only running a timing'
echo "2 cores"
mpirun -n 2 ./fuzzyMPIMain1
echo "4 cores"
mpirun -n 4 ./fuzzyMPIMain1
echo "8 cores"
mpirun -n 8 ./fuzzyMPIMain1
echo "16 cores"
mpirun -n 16 ./fuzzyMPIMain1
echo "32 cores"
mpirun -n 32 ./fuzzyMPIMain1
echo "64 cores"
mpirun -n 64 ./fuzzyMPIMain1
echo "128 cores"
mpirun -n 128 ./fuzzyMPIMain1
# mpirun -n 256 ./fuzzyMPIMain1