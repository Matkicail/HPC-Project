CC = gcc
DEP = -lm

all: clean main cuda mpi

main:
	gcc main.c -o main $(DEP)

cuda:
	nvcc -Xcompiler -fopenmp -I "./inc" main.cu -o cuda

mpi:
	mpicc mainMPI.c -o mpi -lm

# Determine if windows or linux
ifeq ($(OS),Windows_NT)
    RM = del main.exe; del cuda.exe
else
    RM = rm -f main; rm -f cuda; rm -f mpi
endif

clean:
	$(RM) 