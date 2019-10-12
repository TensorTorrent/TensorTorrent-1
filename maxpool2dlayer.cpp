// Author: Yuning Jiang
// Date: Oct. 11 th, 2019
// Description: Max pooling 2D layer.

#include "maxpool2dlayer.h"


using namespace ftensor;
using std::cerr;
using std::endl;


MaxPool2dLayer::MaxPool2dLayer(int kernel_size)
: Layer() {
	if (kernel_size > 0) {
		kernel_size_ = kernel_size;
	}
	else {
		cerr << "Error: Invalid kernel size." << endl;
	}
}


MaxPool2dLayer::~MaxPool2dLayer() {
}


Tensor MaxPool2dLayer::Forward(const Tensor& input) {
	return MaxPool2d(input, kernel_size_, &mask_);
}


Tensor MaxPool2dLayer::Backward(const Tensor& gradient) {
	Tensor a;
	return a;
}
