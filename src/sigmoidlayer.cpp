// Author: Yuning Jiang
// Date: Oct. 17 th, 2019
// Description: Sigmoid layer.

#include "sigmoidlayer.h"


using namespace ftensor;


const float EXP_LIMIT = 85.0;


SigmoidLayer::SigmoidLayer()
: Layer() {
}


SigmoidLayer::~SigmoidLayer() {
}


Tensor SigmoidLayer::Forward(const Tensor& input) {
	output_ = Div(1.0, 1.0 + Exp(Where(input < -EXP_LIMIT, EXP_LIMIT, -input)));
	return output_;
}


Tensor SigmoidLayer::Backward(const Tensor& gradient) {
	grad_output_ = Mul(gradient, Mul(output_, 1.0 - output_));
	return grad_output_;
}
