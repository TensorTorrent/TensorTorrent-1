// Author: Yuning Jiang
// Date: Oct. 16 th, 2019
// Description: Batch normalization layer for 2-D cases.

#include "batchnorm2dlayer.h"


using namespace ftensor;


BatchNorm2dLayer::BatchNorm2dLayer(int num_features, float eps)
: Layer() {
	num_features_ = num_features;
	eps_ = eps;
	batch_size_ = 1;
	has_weight_ = true;
	has_bias_ = true;
	w_ = Ones(num_features_, 1);
	dw_ = Zeros(num_features_, 1);
	b_ = Zeros(num_features_, 1);
	db_ = Zeros(num_features_, 1);
	e_input_ = Zeros(num_features_, 1);
	var_input_ = Zeros(num_features_, 1);
}


BatchNorm2dLayer::~BatchNorm2dLayer() {
}


Tensor BatchNorm2dLayer::Forward(const Tensor& input) {
	shape_[0] = input.rows();
	shape_[1] = input.cols();
	shape_[2] = input.slis();
	shape_[3] = input.gros();
	xi_ = Transpose(Reshape(Permute(input, 0, 1, 3, 2), shape_[0] * shape_[1] * shape_[3], shape_[2], 1, 1));
	batch_size_ = xi_.cols();
	e_input_ = Mean(xi_, 1);
	var_input_ = Var(xi_, 1, "0");
	r_e_input_ = Repmat(e_input_, 1, batch_size_, 1, 1);
	r_var_input_ = Repmat(var_input_, 1, batch_size_, 1, 1);
	r_w_ = Repmat(w_, 1, batch_size_, 1, 1);
	r_b_ = Repmat(b_, 1, batch_size_, 1, 1);
	return Permute(Reshape(Transpose(Mul(Div(xi_ - r_e_input_, Sqrt(r_var_input_ + eps_)), r_w_) + r_b_), shape_[0], shape_[1], shape_[3], shape_[2]), 0, 1, 3, 2);
}


Tensor BatchNorm2dLayer::Backward(const Tensor& gradient) {
	Tensor grad = Transpose(Reshape(Permute(gradient, 0, 1, 3, 2), shape_[0] * shape_[1] * shape_[3], shape_[2], 1, 1));
	Tensor output_temp = Transpose(Reshape(Permute(output_, 0, 1, 3, 2), shape_[0] * shape_[1] * shape_[3], shape_[2], 1, 1));
	Tensor dldxh = Mul(grad, r_w_);
	Tensor dldob2 = Sum(Mul(Mul(-0.5 * dldxh, xi_ - r_e_input_), Pow(r_var_input_ + eps_, -1.5)), 1);
	Tensor dldub = Sum(Mul(dldxh, Div(-1.0, Sqrt(r_var_input_ + eps_))), 1) + Mul(dldob2, Sum(-2.0 * (xi_ - r_e_input_), 1));
	Tensor dldxi = Mul(dldxh, Div(1.0, Sqrt(r_var_input_ + eps_))) + Mul(Repmat(dldob2, 1, batch_size_, 1, 1), Mul(xi_ - r_e_input_, 2.0 / batch_size_)) + Mul(Repmat(dldub, 1, batch_size_, 1, 1), 1.0 / batch_size_);
	dw_ = Sum(Mul(grad, output_temp), 1);
	db_ = Sum(grad, 1);
	return Permute(Reshape(Transpose(dldxi), shape_[0], shape_[1], shape_[3], shape_[2]), 0, 1, 3, 2);
}
