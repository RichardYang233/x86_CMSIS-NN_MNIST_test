# ----------------------- #

# Complie set

CC = gcc
AR = ar
CFLAGS = -std=c99 -Wall -g

# Include Dir

INCLUDES = 	-I./CMSIS-NN/Include \
			-I./CMSIS-NN/Include/Internal \
			-I./DataReader \
			-I./NNInference \
			-I.

# Static Library

LIBRARY = ./libcmsis-nn.a \

# Source files

SRCS = main.c \
       ./DataReader/params_reader.c \
	   ./DataReader/test_dataset_reader.c \
	   ./DataReader/file_utils.c \
	   ./DataReader/result_evaluate.c \
	   ./NNInference/NNInference.c


# Object files

OBJS = $(SRCS:.c=.o)

# Output executable

TARGET = main.exe

# ------------------------ #

.PHONY: all run clean

all: run clean

run: $(TARGET)
	@echo "Running..."
	.\$(TARGET) \

$(TARGET): $(LIBRARY) $(OBJS)
	@echo "Linking..."
	$(CC) $(CFLAGS) -o $@ $^ -L.. $<

%.o: %.c
	@echo "Compiling $<... "
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean: 
	del /s /q $(TARGET) *.o > nul






