CC = gcc
DEP = -lm

all: main

main:
	gcc main.c -o main $(DEP)


# Determine if windows or linux
ifeq ($(OS),Windows_NT)
    RM = del main.exe
else
    RM = rm -f main
endif

clean:
	$(RM) 