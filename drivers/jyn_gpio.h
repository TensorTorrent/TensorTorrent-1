// Copyright 2019 PKU
// Author: Yuning Jiang
// Date: Sep. 24th, 2019
// Description: GPIO control using Linux drivers

#ifndef JYN_GPIO_H_
#define JYN_GPIO_H_

#include <cstdio>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/select.h>
#include <sys/stat.h>


/* returns -1 or the file descriptor of the gpio value file */
int gpio_export(int gpio);

/* Set direction to 2 = high output, 1 low output, 0 input */
int gpio_direction(int gpio, int dir);

/* Release the GPIO to be claimed by other processes or a kernel driver */
void gpio_unexport(int gpio);

/* Single GPIO read */
int gpio_read(int gpio);

/* Set GPIO to val (1 = high) */
int gpio_write(int gpio, int val);

/* Set which edge(s) causes the value select to return */
int gpio_setedge(int gpio, int rising, int falling);

/* Blocks on select until GPIO toggles on edge */
int gpio_select(int gpio);

/* Return the GPIO file descriptor */
int gpio_getfd(int gpio);


#endif  // JYN_GPIO_H_
