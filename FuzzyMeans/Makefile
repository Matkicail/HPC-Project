CC = gcc
DEP = -lm

all: main

main:
	gcc fuzzyMain.c -o fuzzyMain $(DEP)


# Determine if windows or linux
ifeq ($(OS),Windows_NT)
    RM = del fuzzyMain.exe
else
    RM = rm -f fuzzyMain
endif

clean:
	$(RM) 