CC = gcc
CFLAGS = 
DEP = -lm

default: main

main:
	$(CC) $(CFLAGS) main.c $(DEP) -o main
	chmod a+x main

clean:
	rm main.exe