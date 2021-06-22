CC = gcc
DEP = -lm

all: clean main cuda

main:
	gcc fuzzyMain.c -o fuzzyMain $(DEP)

cuda:
	nvcc -Xcompiler -fopenmp -I "./inc" fuzzyMain.cu -o fuzzyCuda --gpu-architecture=compute_75

# Determine if windows or linux
ifeq ($(OS),Windows_NT)
    RM = del main.exe; del cuda.exe
else
    RM = rm -f fuzzyMain; rm -f fuzzyCuda
endif

clean:
	$(RM) 