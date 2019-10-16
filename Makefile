TARGET = ${EX_DIR}/lenet-5

MAIN_DIR = .
DRIVER_DIR = ${MAIN_DIR}/drivers
SRC_DIR = ${MAIN_DIR}/src
EX_DIR = ${MAIN_DIR}/examples
INC_DIR= -I${MAIN_DIR} \
	-I${SRC_DIR} \
	-I${DRIVER_DIR}


src1 = $(wildcard ${SRC_DIR}/*.cpp)
src2 = $(wildcard ${EX_DIR}/*.cpp)
obj1 = $(patsubst %.cpp, %.o, $(src1))
obj2 = $(patsubst %.cpp, %.o, $(src2))
obj3 = drivers/jyn_spi.co drivers/jyn_gpio.o drivers/jyn_flash.o


CFLAGS = -g -Wall -std=c++11 -O1 ${INC_DIR}
LDFLAGS = -g -Wall -std=c++11 -O1
CLDFLAGS = -g -Wall -O1
CPP = g++
CC = gcc
#CPP = arm-linux-gnueabihf-g++


build: $(TARGET)

$(TARGET): $(obj1) $(obj2) ${obj3}
	 ${CPP} $(LDFLAGS) $^ -o $@ 

%.o : %.cpp
	$(CPP) $(CFLAGS) -c $< -o $@
%.o:$(SRC_DIR)/%.cpp
	$(CPP) $(CFLAGS) -c $< -o $@
%.o:$(EX_DIR)/%.cpp
	$(CPP) $(CFLAGS) -c $< -o $@

drivers/jyn_spi.co : drivers/jyn_spi.c drivers/jyn_spi.h
	$(CC) $(CLDFLAGS) -c $< -o $@
drivers/jyn_gpio.o : drivers/jyn_gpio.cpp drivers/jyn_gpio.h
	$(CPP) -g -std=c++11 -O1 -c $< -o $@
drivers/jyn_flash.o : drivers/jyn_flash.cpp drivers/jyn_flash.h drivers/jyn_spi.h drivers/jyn_spi.co
	$(CPP) $(CFLAGS) -c $< -o $@

edit : $(TARGET)
	cc -o edit $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET) ${MAIN_DIR}/*.o ${SRC_DIR}/*.o ${EX_DIR}/*.o ${DRIVER_DIR}/*.co ${DRIVER_DIR}/*.o
