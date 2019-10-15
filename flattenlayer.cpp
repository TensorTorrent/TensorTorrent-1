// Author: Yuning Jiang
// Date: Oct. 11 th, 2019
// Description: Flatten layer.

#include "flattenlayer.h"


using namespace ftensor;


FlattenLayer::FlattenLayer(int start_dim, int end_dim)
: Layer() {
	start_dim_ = start_dim;
	end_dim_ = end_dim;
	shape_[0] = 0;
	shape_[1] = 0;
	shape_[2] = 0;
	shape_[3] = 0;
}


FlattenLayer::~FlattenLayer() {
}


Tensor FlattenLayer::Forward(const Tensor& input) {
	shape_[0] = input.rows();
	shape_[1] = input.cols();
	shape_[2] = input.slis();
	shape_[3] = input.gros();
	output_ = Transpose(Reshape(input, shape_[3], shape_[0] * shape_[1] * shape_[2], 1, 1));
	return output_;
}


Tensor FlattenLayer::Backward(const Tensor& gradient) {
	grad_output_ = Reshape(Transpose(gradient), shape_[0], shape_[1], shape_[2], shape_[3]);
	return grad_output_;
}
