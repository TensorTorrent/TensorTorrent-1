// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: Convolution layer.

#include "convtranspose2dlayer.h"


ConvTranspose2dLayer::ConvTranspose2dLayer(int in_channels, int out_channels, int kernel_size, int stride, int padding) {
	in_channels_ = in_channels;
	out_channels_ = out_channels;
	kernel_size_ = kernel_size;
	stride_ = stride;
	padding_ = padding;
}


ConvTranspose2dLayer::~ConvTranspose2dLayer() {
}
