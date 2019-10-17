// Author: Yuning Jiang
// Date: Oct. 11 th, 2019
// Description: ReLU layer.

#include "relulayer.h"


using namespace ftensor;


ReluLayer::ReluLayer()
: Layer() {
}


ReluLayer::~ReluLayer() {
}


Tensor ReluLayer::Forward(const Tensor& input) {
	output_ = Where(input > 0, input, 0);
	return output_;
}


Tensor ReluLayer::Backward(const Tensor& gradient) {
	grad_output_ = Where(output_ > 0, gradient, 0);
	return grad_output_;
}
