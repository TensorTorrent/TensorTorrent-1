// Author: Yuning Jiang
// Date: Oct. 17 th, 2019
// Description: Tanh layer.

#include "tanhlayer.h"


using namespace ftensor;


const float EXP_LIMIT = 85.0;


TanhLayer::TanhLayer()
: Layer() {
}


TanhLayer::~TanhLayer() {
}


Tensor TanhLayer::Forward(const Tensor& input) {
	output_ = Tanh(Where(input > EXP_LIMIT, EXP_LIMIT, input));
	return output_;
}


Tensor TanhLayer::Backward(const Tensor& gradient) {
	grad_output_ = Mul(gradient, 1.0 - Pow(output_, 2.0));
	return grad_output_;
}
