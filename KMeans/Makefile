CC = gcc
DEP = -lm

all: clean main cuda

main:
	gcc main.c -o main $(DEP)

cuda:
	nvcc -I "./inc" main.cu -o cuda

# Determine if windows or linux
ifeq ($(OS),Windows_NT)
    RM = del main.exe; del cuda.exe
else
    RM = rm -f main; rm -f cuda
endif

clean:
	$(RM) 