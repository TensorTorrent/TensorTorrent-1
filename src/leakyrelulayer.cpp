// Author: Yuning Jiang
// Date: Oct. 17 th, 2019
// Description: Leaky ReLU layer.

#include "leakyrelulayer.h"


using namespace ftensor;


LeakyReluLayer::LeakyReluLayer(float negative_slope)
: Layer() {
	negative_slope_ = negative_slope;
}


LeakyReluLayer::~LeakyReluLayer() {
}


Tensor LeakyReluLayer::Forward(const Tensor& input) {
	output_ = Where(input >= 0, input, negative_slope_ * input);
	return output_;
}


Tensor LeakyReluLayer::Backward(const Tensor& gradient) {
	grad_output_ = Where(output_ >= 0, gradient, negative_slope_ * gradient);
	return grad_output_;
}
