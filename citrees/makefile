CFLAGS  = -fPIC -Ofast -march=native -ffast-math 
LDFLAGS = -shared
CC      = gcc

bin/dcor.so: src/dcor.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o bin/dcor.so src/dcor.c

clean:
	rm -f bin/*so