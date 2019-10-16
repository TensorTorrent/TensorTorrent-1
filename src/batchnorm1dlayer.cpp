// Author: Yuning Jiang
// Date: Oct. 16 th, 2019
// Description: Batch normalization layer for 1-D cases.

#include "batchnorm1dlayer.h"


using namespace ftensor;


BatchNorm1dLayer::BatchNorm1dLayer(int num_features, float eps)
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


BatchNorm1dLayer::~BatchNorm1dLayer() {
}


Tensor BatchNorm1dLayer::Forward(const Tensor& input) {
	xi_ = input;
	batch_size_ = input.cols();
	e_input_ = Mean(input, 1);
	var_input_ = Var(input, 1, "0");
	r_e_input_ = Repmat(e_input_, 1, batch_size_, 1, 1);
	r_var_input_ = Repmat(var_input_, 1, batch_size_, 1, 1);
	r_w_ = Repmat(w_, 1, batch_size_, 1, 1);
	r_b_ = Repmat(b_, 1, batch_size_, 1, 1);
	return Mul(Div(input - r_e_input_, Sqrt(r_var_input_ + eps_)), r_w_) + r_b_;
}


Tensor BatchNorm1dLayer::Backward(const Tensor& gradient) {
	Tensor dldxh = Mul(gradient, r_w_);
	Tensor dldob2 = Sum(Mul(Mul(-0.5 * dldxh, xi_ - r_e_input_), Pow(r_var_input_ + eps_, -1.5)), 1);
	Tensor dldub = Sum(Mul(dldxh, Div(-1.0, Sqrt(r_var_input_ + eps_))), 1) + Mul(dldob2, Sum(-2.0 * (xi_ - r_e_input_), 1));
	Tensor dldxi = Mul(dldxh, Div(1.0, Sqrt(r_var_input_ + eps_))) + Mul(Repmat(dldob2, 1, batch_size_, 1, 1), Mul(xi_ - r_e_input_, 2.0 / batch_size_)) + Mul(Repmat(dldub, 1, batch_size_, 1, 1), 1.0 / batch_size_);
	dw_ = Sum(Mul(gradient, output_), 1);
	db_ = Sum(gradient, 1);
	return dldxi;
}
