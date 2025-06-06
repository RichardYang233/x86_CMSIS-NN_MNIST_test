
# ------------------------------------------------------------------------
#
# ------------------------------------------------------------------------

# compile set

CC = gcc
CFLAGS = -std=c11 -Wall -g

# static lib

LIBRARY = ./libcmsis-nn.a					\

# include

INCLUDES = 	-I./CMSIS-NN/Include 			\
			-I./CMSIS-NN/Include/Internal 	\
			-I./model						\
			-I./model/FCNet					\
			-I./dataset						\
			-I./utils						\
			-I.

# source

SRCS = 	./main.c								\
		./utils/cmsis_nn_helper.c

# 

OBJS = $(SRCS:.c=.o)

TARGET = main.exe

# ------------------------------------------------------------------------
#
# ------------------------------------------------------------------------

.PHONY: all run clean

all: run clean

run: $(TARGET)
	@echo "Running..."
	@.\$(TARGET)

$(TARGET): $(LIBRARY) $(OBJS)
	@echo "Linking..."
	@$(CC) $(CFLAGS) -o $@ $^ -L.. $<

%.o: %.c
	@echo "Compiling $<... "
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean: 
	@del /s /q $(TARGET) *.o > nul  




