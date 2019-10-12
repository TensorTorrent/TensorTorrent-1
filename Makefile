TARGET = main

MAIN_DIR = .
DRIVER_DIR = ${MAIN_DIR}/drivers
INC_DIR= -I${MAIN_DIR} \
	-I${DRIVER_DIR}

CFLAGS = -g -Wall -std=c++11 -O1 ${INC_DIR}
LDFLAGS = -g -Wall -std=c++11 -O1
CLDFLAGS = -g -Wall -O1
CPP = g++
CC = gcc
#CPP = arm-linux-gnueabihf-g++


build: $(TARGET)

$(TARGET): main.o tensorlib.o device.o layer.o conv2dlayer.o convtranspose2dlayer.o mnistloader.o \
	flattenlayer.o relulayer.o softmaxlayer.o linearlayer.o maxpool2dlayer.o sequential.o
# goku.o conv2dgoku.o convtranspose2dgoku.o drivers/jyn_spi.co drivers/jyn_gpio.o drivers/jyn_flash.o
	 ${CPP} $(LDFLAGS) $^ -o $@ 

%.o : %.cpp
	$(CPP) $(CFLAGS) -c $< -o $@

#drivers/jyn_spi.co : drivers/jyn_spi.c drivers/jyn_spi.h
#	$(CC) $(CLDFLAGS) -c $< -o $@
#drivers/jyn_gpio.o : drivers/jyn_gpio.cpp drivers/jyn_gpio.h
#	$(CPP) -g -std=c++11 -O1 -c $< -o $@
#drivers/jyn_flash.o : drivers/jyn_flash.cpp drivers/jyn_flash.h drivers/jyn_spi.h drivers/jyn_spi.co
#	$(CPP) $(CFLAGS) -c $< -o $@

edit : $(TARGET)
	cc -o edit $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET) ${MAIN_DIR}/*.o ${DRIVER_DIR}/*.co ${DRIVER_DIR}/*.o
