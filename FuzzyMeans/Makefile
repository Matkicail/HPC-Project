CC = gcc
DEP = -lm

all: clean main cuda mpi

main:
	gcc fuzzyMain.c -o fuzzyMain $(DEP)

cuda:
	nvcc -Xcompiler -fopenmp -I "./inc" fuzzyMain.cu -o fuzzyCuda25 --gpu-architecture=compute_75

mpi:
	mpicc fuzzyMPIMain.c -o fuzzyMPIMain1 -lm

# Determine if windows or linux
ifeq ($(OS),Windows_NT)
    RM = del main.exe; del cuda.exe; del fuzzyMPIMain
else
    RM = rm -f fuzzyMain; rm -f fuzzyCuda; rm -f fuzzyMPIMain
endif

clean:
	$(RM) 