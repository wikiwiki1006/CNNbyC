CC = gcc
CFLAGS = -Wall -O2 -I./include
SRC = $(wildcard src/*.c)
TARGET = mnist_reader

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)