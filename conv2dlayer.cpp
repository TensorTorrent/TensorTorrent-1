// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: Convolution layer.

#include "conv2dlayer.h"


using namespace ftensor;
using std::endl;
using std::cerr;


Conv2dLayer::Conv2dLayer(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool bias)
: Layer() {
	in_channels_ = in_channels;
	out_channels_ = out_channels;
	kernel_size_ = kernel_size;
	stride_ = stride;
	padding_ = padding;
	init_range_ = sqrt(1.0 / in_channels_ / kernel_size_ / kernel_size_);
	kernel_ = Rand(kernel_size_, kernel_size_, in_channels_, out_channels_, init_range_, -init_range_);
	using_bias_ = bias;
	if (using_bias_) {
		bias_ = Rand(1, 1, 1, 1, init_range_, -init_range_);
	}
}


Conv2dLayer::~Conv2dLayer() {
}


Tensor Conv2dLayer::Forward(const Tensor& input_image) {
	return Conv2d(input_image, kernel_, stride_, padding_);
}


Tensor Conv2dLayer::Backward(const Tensor& gradient) {
	Tensor a;
	return a;
}


void Conv2dLayer::SetWeight(const ftensor::Tensor& kernel) {
	if (Match(kernel, kernel_)) {
		kernel_ = kernel;
	}
	else {
		cerr << "Error: Kernel size mismatched." << endl;
		exit(1);
	}
}
