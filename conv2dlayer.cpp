// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: Convolution layer.

#include "conv2dlayer.h"


Conv2dLayer::Conv2dLayer(int in_channels, int out_channels, int kernel_size, int stride, int padding) {
	in_channels_ = in_channels;
	out_channels_ = out_channels;
	kernel_size_ = kernel_size;
	stride_ = stride;
	padding_ = padding;
}


Conv2dLayer::~Conv2dLayer() {
}
