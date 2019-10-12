// Author: Yuning Jiang
// Date: Oct. 11 th, 2019
// Description: Linear layer.

#include "linearlayer.h"


using namespace ftensor;
using std::cerr;
using std::endl;


LinearLayer::LinearLayer(int in_features, int out_features, bool bias)
: Layer() {
	in_features_ = in_features;
	out_features_ = out_features;
	init_range_ = sqrt(1.0 / in_features_);
	weights_ = Rand(out_features_, in_features_, 1, 1, init_range_, -init_range_);
	dw_ = Zeros(weights_);
	using_bias_ = bias;
	if (using_bias_) {
		bias_ = Rand(1, 1, 1, 1, init_range_, -init_range_);
	}
}


LinearLayer::~LinearLayer() {
}


Tensor LinearLayer::Forward(const Tensor& input_image) {
	output_ = MM(weights_, input_image);
	return output_;
}


Tensor LinearLayer::Backward(const Tensor& gradient) {
	if (is_first_layer_) {
		dw_ += MM(gradient, Transpose(input_));
	}
	else {
		dw_ += MM(gradient, Transpose(previous_layer_->GetOutput()));
	}
	grad_output_ = MM(Transpose(weights_), gradient);
	return grad_output_;
}


void LinearLayer::SetWeight(const ftensor::Tensor& weights) {
	if (Match(weights, weights_)) {
		weights_ = weights;
	}
	else {
		cerr << "Error: Weight size mismatched." << endl;
		exit(1);
	}
}


void LinearLayer::ZeroGrad() {
	dw_.Zeros();
}
