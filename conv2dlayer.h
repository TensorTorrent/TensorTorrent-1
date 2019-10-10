// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: Convolution layer.

#ifndef __CONV2D_LAYER_H__
#define __CONV2D_LAYER_H__


#include <iostream>

#include "tensorlib.h"
#include "layer.h"


class Conv2dLayer {
public:
	Conv2dLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0);
	virtual ~Conv2dLayer();

protected:
	int in_channels_;
	int out_channels_;
	int kernel_size_;
	int stride_;
	int padding_;
};


#endif  // __CONV2D_LAYER_H__
