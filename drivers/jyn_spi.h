// Copyright 2019 PKU
// Author: Yuning Jiang
// Date: Sep. 24th, 2019
// Description: SPI control using Linux drivers

#ifndef JYN_SPI_H_
#define JYN_SPI_H_

#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <linux/spi/spidev.h>


#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define DEFAULT_SPI_MODE		0
#define DEFAULT_SPI_SPEED		50000000
#define DEFAULT_SPI_BITS		8
#define DEFAULT_SPI_DELAY		0

#ifdef TPU
#define DEFAULT_DEVICE "/dev/spidev32766.1"
#else
#define DEFAULT_DEVICE "/dev/spidev0.0"
#endif




#endif  // JYN_SPI_H_
